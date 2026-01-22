//! Multi-tenant memory isolation
//!
//! This module provides tenant-based isolation for the memory pool,
//! allowing multiple tenants to share the same pool with resource limits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use super::page::PageId;

/// Unique identifier for a tenant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantId(pub u32);

impl TenantId {
    /// Create a new tenant ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// System tenant (ID 0)
    pub const SYSTEM: TenantId = TenantId(0);
}

impl From<u32> for TenantId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tenant({})", self.0)
    }
}

/// Configuration for a tenant
#[derive(Debug, Clone)]
pub struct TenantConfig {
    /// Tenant ID
    pub id: TenantId,
    /// Tenant name (for logging)
    pub name: String,
    /// Maximum pages this tenant can allocate
    pub max_pages: usize,
    /// Reserved pages (guaranteed minimum)
    pub reserved_pages: usize,
    /// Priority (higher = more important during eviction)
    pub priority: u8,
    /// Whether tenant can use shared pages
    pub can_share: bool,
}

impl TenantConfig {
    /// Create a new tenant configuration
    pub fn new(id: TenantId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            max_pages: usize::MAX,
            reserved_pages: 0,
            priority: 5,
            can_share: true,
        }
    }

    /// Set maximum pages
    pub fn max_pages(mut self, pages: usize) -> Self {
        self.max_pages = pages;
        self
    }

    /// Set reserved pages
    pub fn reserved_pages(mut self, pages: usize) -> Self {
        self.reserved_pages = pages;
        self
    }

    /// Set priority
    pub fn priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Runtime state for a tenant
struct TenantState {
    /// Configuration
    config: TenantConfig,
    /// Currently allocated pages
    allocated_pages: AtomicU64,
    /// Number of active allocations
    active_allocations: AtomicU32,
    /// Total allocations (lifetime)
    total_allocations: AtomicU64,
    /// Total deallocations (lifetime)
    total_deallocations: AtomicU64,
}

impl TenantState {
    fn new(config: TenantConfig) -> Self {
        Self {
            config,
            allocated_pages: AtomicU64::new(0),
            active_allocations: AtomicU32::new(0),
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
        }
    }

    fn try_allocate(&self, pages: usize) -> bool {
        let current = self.allocated_pages.load(Ordering::Acquire);
        let new_total = current + pages as u64;

        if new_total > self.config.max_pages as u64 {
            return false;
        }

        // Try to increment
        match self.allocated_pages.compare_exchange(
            current,
            new_total,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.active_allocations.fetch_add(1, Ordering::Relaxed);
                self.total_allocations.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                // Retry
                self.try_allocate(pages)
            }
        }
    }

    fn deallocate(&self, pages: usize) {
        self.allocated_pages
            .fetch_sub(pages as u64, Ordering::Relaxed);
        self.active_allocations.fetch_sub(1, Ordering::Relaxed);
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }

    fn stats(&self) -> TenantStats {
        TenantStats {
            id: self.config.id,
            name: self.config.name.clone(),
            allocated_pages: self.allocated_pages.load(Ordering::Relaxed) as usize,
            max_pages: self.config.max_pages,
            reserved_pages: self.config.reserved_pages,
            active_allocations: self.active_allocations.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            priority: self.config.priority,
        }
    }
}

/// Statistics for a tenant
#[derive(Debug, Clone)]
pub struct TenantStats {
    /// Tenant ID
    pub id: TenantId,
    /// Tenant name
    pub name: String,
    /// Currently allocated pages
    pub allocated_pages: usize,
    /// Maximum allowed pages
    pub max_pages: usize,
    /// Reserved pages
    pub reserved_pages: usize,
    /// Active allocations
    pub active_allocations: u32,
    /// Total allocations (lifetime)
    pub total_allocations: u64,
    /// Total deallocations (lifetime)
    pub total_deallocations: u64,
    /// Priority level
    pub priority: u8,
}

impl TenantStats {
    /// Get utilization ratio
    pub fn utilization(&self) -> f64 {
        if self.max_pages == 0 || self.max_pages == usize::MAX {
            0.0
        } else {
            self.allocated_pages as f64 / self.max_pages as f64
        }
    }

    /// Get available pages
    pub fn available_pages(&self) -> usize {
        self.max_pages.saturating_sub(self.allocated_pages)
    }
}

/// Manager for multi-tenant memory isolation
pub struct TenantManager {
    /// Tenant states
    tenants: RwLock<HashMap<TenantId, Arc<TenantState>>>,
    /// Next tenant ID
    next_id: AtomicU32,
    /// Total pages in the pool
    total_pool_pages: usize,
}

impl TenantManager {
    /// Create a new tenant manager
    pub fn new(total_pool_pages: usize) -> Self {
        let mut tenants = HashMap::new();

        // Create system tenant
        let system_config = TenantConfig::new(TenantId::SYSTEM, "system").priority(10); // Highest priority
        tenants.insert(TenantId::SYSTEM, Arc::new(TenantState::new(system_config)));

        Self {
            tenants: RwLock::new(tenants),
            next_id: AtomicU32::new(1),
            total_pool_pages,
        }
    }

    /// Register a new tenant
    pub fn register_tenant(&self, config: TenantConfig) -> TenantId {
        let id = if config.id == TenantId::SYSTEM {
            // Auto-assign ID
            TenantId::new(self.next_id.fetch_add(1, Ordering::Relaxed))
        } else {
            config.id
        };

        let state = TenantState::new(TenantConfig { id, ..config });

        let mut tenants = self.tenants.write();
        tenants.insert(id, Arc::new(state));

        id
    }

    /// Unregister a tenant
    pub fn unregister_tenant(&self, id: TenantId) -> bool {
        if id == TenantId::SYSTEM {
            return false; // Cannot unregister system tenant
        }

        let mut tenants = self.tenants.write();
        tenants.remove(&id).is_some()
    }

    /// Get tenant state
    fn get_tenant(&self, id: TenantId) -> Option<Arc<TenantState>> {
        let tenants = self.tenants.read();
        tenants.get(&id).cloned()
    }

    /// Try to allocate pages for a tenant
    pub fn try_allocate(&self, tenant_id: TenantId, pages: usize) -> bool {
        if let Some(state) = self.get_tenant(tenant_id) {
            state.try_allocate(pages)
        } else {
            false
        }
    }

    /// Record deallocation for a tenant
    pub fn deallocate(&self, tenant_id: TenantId, pages: usize) {
        if let Some(state) = self.get_tenant(tenant_id) {
            state.deallocate(pages);
        }
    }

    /// Get tenant statistics
    pub fn tenant_stats(&self, id: TenantId) -> Option<TenantStats> {
        self.get_tenant(id).map(|s| s.stats())
    }

    /// Get all tenant statistics
    pub fn all_stats(&self) -> Vec<TenantStats> {
        let tenants = self.tenants.read();
        tenants.values().map(|s| s.stats()).collect()
    }

    /// Get tenant priority for eviction decisions
    pub fn tenant_priority(&self, id: TenantId) -> u8 {
        self.get_tenant(id).map(|s| s.config.priority).unwrap_or(0)
    }

    /// Check if tenant exists
    pub fn tenant_exists(&self, id: TenantId) -> bool {
        let tenants = self.tenants.read();
        tenants.contains_key(&id)
    }

    /// Get number of registered tenants
    pub fn tenant_count(&self) -> usize {
        let tenants = self.tenants.read();
        tenants.len()
    }

    /// Calculate fair share for a tenant
    pub fn fair_share(&self, id: TenantId) -> usize {
        let tenants = self.tenants.read();
        let count = tenants.len();
        if count == 0 {
            return 0;
        }

        // Simple fair share: total / count
        // Could be enhanced with priority-based weighting
        self.total_pool_pages / count
    }
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new(4096) // Default pool size
    }
}

/// Tenant-scoped allocation guard
pub struct TenantAllocation {
    manager: Arc<TenantManager>,
    tenant_id: TenantId,
    pages: usize,
}

impl TenantAllocation {
    /// Create a new tenant allocation
    pub fn new(manager: Arc<TenantManager>, tenant_id: TenantId, pages: usize) -> Option<Self> {
        if manager.try_allocate(tenant_id, pages) {
            Some(Self {
                manager,
                tenant_id,
                pages,
            })
        } else {
            None
        }
    }

    /// Get the tenant ID
    pub fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }

    /// Get the number of pages
    pub fn pages(&self) -> usize {
        self.pages
    }
}

impl Drop for TenantAllocation {
    fn drop(&mut self) {
        self.manager.deallocate(self.tenant_id, self.pages);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id() {
        let id = TenantId::new(42);
        assert_eq!(id.raw(), 42);
        assert_eq!(TenantId::SYSTEM, TenantId::new(0));
    }

    #[test]
    fn test_tenant_config() {
        let config = TenantConfig::new(TenantId::new(1), "test")
            .max_pages(100)
            .reserved_pages(10)
            .priority(8);

        assert_eq!(config.max_pages, 100);
        assert_eq!(config.reserved_pages, 10);
        assert_eq!(config.priority, 8);
    }

    #[test]
    fn test_tenant_manager_basic() {
        let manager = TenantManager::new(1000);

        // System tenant should exist
        assert!(manager.tenant_exists(TenantId::SYSTEM));

        // Register new tenant
        let config = TenantConfig::new(TenantId::new(1), "tenant1").max_pages(100);
        let id = manager.register_tenant(config);

        assert!(manager.tenant_exists(id));
        assert_eq!(manager.tenant_count(), 2);
    }

    #[test]
    fn test_tenant_allocation() {
        let manager = TenantManager::new(1000);

        let config = TenantConfig::new(TenantId::new(1), "tenant1").max_pages(100);
        let id = manager.register_tenant(config);

        // Allocate within limits
        assert!(manager.try_allocate(id, 50));

        let stats = manager.tenant_stats(id).unwrap();
        assert_eq!(stats.allocated_pages, 50);

        // Allocate more within limits
        assert!(manager.try_allocate(id, 40));

        // Exceed limits
        assert!(!manager.try_allocate(id, 20));

        // Deallocate
        manager.deallocate(id, 50);
        let stats = manager.tenant_stats(id).unwrap();
        assert_eq!(stats.allocated_pages, 40);
    }

    #[test]
    fn test_tenant_unregister() {
        let manager = TenantManager::new(1000);

        let config = TenantConfig::new(TenantId::new(1), "tenant1");
        let id = manager.register_tenant(config);

        assert!(manager.unregister_tenant(id));
        assert!(!manager.tenant_exists(id));

        // Cannot unregister system tenant
        assert!(!manager.unregister_tenant(TenantId::SYSTEM));
    }

    #[test]
    fn test_tenant_stats() {
        let manager = TenantManager::new(1000);

        let config = TenantConfig::new(TenantId::new(1), "tenant1").max_pages(100);
        let id = manager.register_tenant(config);

        manager.try_allocate(id, 30);
        manager.try_allocate(id, 20);
        manager.deallocate(id, 20);

        let stats = manager.tenant_stats(id).unwrap();
        assert_eq!(stats.allocated_pages, 30);
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.active_allocations, 1);
    }

    #[test]
    fn test_tenant_allocation_guard() {
        let manager = Arc::new(TenantManager::new(1000));

        let config = TenantConfig::new(TenantId::new(1), "tenant1").max_pages(100);
        let id = manager.register_tenant(config);

        {
            let _alloc = TenantAllocation::new(Arc::clone(&manager), id, 50).unwrap();
            let stats = manager.tenant_stats(id).unwrap();
            assert_eq!(stats.allocated_pages, 50);
        }

        // Should be deallocated on drop
        let stats = manager.tenant_stats(id).unwrap();
        assert_eq!(stats.allocated_pages, 0);
    }

    #[test]
    fn test_fair_share() {
        let manager = TenantManager::new(1000);

        // With just system tenant
        let share = manager.fair_share(TenantId::SYSTEM);
        assert_eq!(share, 1000);

        // Add more tenants
        manager.register_tenant(TenantConfig::new(TenantId::new(1), "t1"));
        manager.register_tenant(TenantConfig::new(TenantId::new(2), "t2"));

        let share = manager.fair_share(TenantId::SYSTEM);
        assert_eq!(share, 333); // 1000 / 3
    }
}
