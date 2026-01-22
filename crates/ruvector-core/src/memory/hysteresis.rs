//! Hysteresis mechanism for thrash prevention
//!
//! This module implements hysteresis control to prevent oscillation between
//! eviction and allocation cycles (thrashing).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Configuration for hysteresis control
#[derive(Debug, Clone)]
pub struct HysteresisConfig {
    /// High watermark - start eviction when utilization exceeds this
    pub high_watermark: f64,
    /// Low watermark - stop eviction when utilization drops below this
    pub low_watermark: f64,
    /// Hysteresis factor - extra pages to evict beyond immediate need
    pub hysteresis_factor: f64,
    /// Minimum eviction batch size
    pub min_batch_size: usize,
    /// Cooldown period after eviction (microseconds)
    pub cooldown_us: u64,
}

impl Default for HysteresisConfig {
    fn default() -> Self {
        Self {
            high_watermark: 0.90,
            low_watermark: 0.80,
            hysteresis_factor: 0.1,
            min_batch_size: 4,
            cooldown_us: 1000, // 1ms
        }
    }
}

impl HysteresisConfig {
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.low_watermark >= self.high_watermark {
            return Err("Low watermark must be less than high watermark");
        }
        if self.high_watermark > 1.0 || self.low_watermark < 0.0 {
            return Err("Watermarks must be between 0.0 and 1.0");
        }
        if self.hysteresis_factor < 0.0 || self.hysteresis_factor > 1.0 {
            return Err("Hysteresis factor must be between 0.0 and 1.0");
        }
        Ok(())
    }

    /// Create a conservative configuration (less aggressive eviction)
    pub fn conservative() -> Self {
        Self {
            high_watermark: 0.95,
            low_watermark: 0.85,
            hysteresis_factor: 0.15,
            min_batch_size: 8,
            cooldown_us: 5000,
        }
    }

    /// Create an aggressive configuration (more aggressive eviction)
    pub fn aggressive() -> Self {
        Self {
            high_watermark: 0.85,
            low_watermark: 0.70,
            hysteresis_factor: 0.05,
            min_batch_size: 2,
            cooldown_us: 500,
        }
    }
}

/// State of the hysteresis controller
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HysteresisState {
    /// Normal operation, no eviction pressure
    Normal,
    /// Eviction in progress
    Evicting,
    /// Cooldown after eviction
    Cooldown,
}

/// Controller for hysteresis-based eviction decisions
pub struct HysteresisController {
    /// Configuration
    config: HysteresisConfig,
    /// Current state
    state: AtomicU8Wrapper,
    /// Last eviction timestamp (microseconds)
    last_eviction: AtomicU64,
    /// Total pages evicted in current cycle
    evicted_in_cycle: AtomicU64,
    /// Whether we're in an eviction cycle
    in_eviction_cycle: AtomicBool,
}

// Wrapper for atomic state storage
#[repr(transparent)]
struct AtomicU8Wrapper(std::sync::atomic::AtomicU8);

impl AtomicU8Wrapper {
    fn new(state: HysteresisState) -> Self {
        Self(std::sync::atomic::AtomicU8::new(state as u8))
    }

    fn load(&self) -> HysteresisState {
        match self.0.load(Ordering::Acquire) {
            0 => HysteresisState::Normal,
            1 => HysteresisState::Evicting,
            _ => HysteresisState::Cooldown,
        }
    }

    fn store(&self, state: HysteresisState) {
        self.0.store(state as u8, Ordering::Release);
    }
}

impl HysteresisController {
    /// Create a new hysteresis controller
    pub fn new(config: HysteresisConfig) -> Self {
        Self {
            config,
            state: AtomicU8Wrapper::new(HysteresisState::Normal),
            last_eviction: AtomicU64::new(0),
            evicted_in_cycle: AtomicU64::new(0),
            in_eviction_cycle: AtomicBool::new(false),
        }
    }

    /// Get the current state
    pub fn state(&self) -> HysteresisState {
        self.state.load()
    }

    /// Check if eviction should be triggered based on utilization
    pub fn should_evict(&self, utilization: f64) -> bool {
        let state = self.state.load();

        match state {
            HysteresisState::Normal => {
                // Start eviction if above high watermark
                utilization > self.config.high_watermark
            }
            HysteresisState::Evicting => {
                // Continue eviction until below low watermark
                utilization > self.config.low_watermark
            }
            HysteresisState::Cooldown => {
                // Check if cooldown has expired
                let now = current_timestamp_us();
                let last = self.last_eviction.load(Ordering::Acquire);
                if now - last > self.config.cooldown_us {
                    self.state.store(HysteresisState::Normal);
                    utilization > self.config.high_watermark
                } else {
                    false
                }
            }
        }
    }

    /// Calculate the eviction target (pages to evict)
    pub fn eviction_target(&self, required_pages: usize) -> usize {
        // Apply hysteresis factor to evict extra pages
        let extra = (required_pages as f64 * self.config.hysteresis_factor).ceil() as usize;
        let target = required_pages + extra;

        // Ensure minimum batch size
        target.max(self.config.min_batch_size)
    }

    /// Notify that eviction is starting
    pub fn start_eviction(&self) {
        self.state.store(HysteresisState::Evicting);
        self.in_eviction_cycle.store(true, Ordering::Release);
        self.evicted_in_cycle.store(0, Ordering::Release);
    }

    /// Notify that pages were evicted
    pub fn record_eviction(&self, count: usize) {
        self.evicted_in_cycle
            .fetch_add(count as u64, Ordering::Relaxed);
        self.last_eviction
            .store(current_timestamp_us(), Ordering::Release);
    }

    /// Notify that eviction cycle is complete
    pub fn end_eviction(&self) {
        self.state.store(HysteresisState::Cooldown);
        self.in_eviction_cycle.store(false, Ordering::Release);
    }

    /// Get the number of pages evicted in the current/last cycle
    pub fn evicted_in_cycle(&self) -> u64 {
        self.evicted_in_cycle.load(Ordering::Acquire)
    }

    /// Check if we're currently in an eviction cycle
    pub fn is_evicting(&self) -> bool {
        self.in_eviction_cycle.load(Ordering::Acquire)
    }

    /// Update configuration
    pub fn configure(&mut self, config: HysteresisConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &HysteresisConfig {
        &self.config
    }
}

/// Get current timestamp in microseconds
fn current_timestamp_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureLevel {
    /// No pressure
    None,
    /// Low pressure
    Low,
    /// Medium pressure
    Medium,
    /// High pressure
    High,
    /// Critical pressure
    Critical,
}

impl PressureLevel {
    /// Determine pressure level from utilization
    pub fn from_utilization(utilization: f64) -> Self {
        match utilization {
            u if u < 0.5 => PressureLevel::None,
            u if u < 0.7 => PressureLevel::Low,
            u if u < 0.85 => PressureLevel::Medium,
            u if u < 0.95 => PressureLevel::High,
            _ => PressureLevel::Critical,
        }
    }

    /// Get recommended eviction factor for this pressure level
    pub fn eviction_factor(self) -> f64 {
        match self {
            PressureLevel::None => 0.0,
            PressureLevel::Low => 0.05,
            PressureLevel::Medium => 0.10,
            PressureLevel::High => 0.20,
            PressureLevel::Critical => 0.30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = HysteresisConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_watermarks() {
        let config = HysteresisConfig {
            high_watermark: 0.7,
            low_watermark: 0.8, // Invalid: low > high
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_controller_normal_state() {
        let controller = HysteresisController::new(HysteresisConfig::default());
        assert_eq!(controller.state(), HysteresisState::Normal);

        // Should not evict below high watermark
        assert!(!controller.should_evict(0.5));
        assert!(!controller.should_evict(0.89));

        // Should evict at high watermark
        assert!(controller.should_evict(0.91));
    }

    #[test]
    fn test_controller_evicting_state() {
        let controller = HysteresisController::new(HysteresisConfig::default());
        controller.start_eviction();

        assert_eq!(controller.state(), HysteresisState::Evicting);

        // Should continue eviction until low watermark
        assert!(controller.should_evict(0.85));
        assert!(controller.should_evict(0.81));

        // Should stop at low watermark
        assert!(!controller.should_evict(0.79));
    }

    #[test]
    fn test_eviction_target() {
        let controller = HysteresisController::new(HysteresisConfig::default());

        // Target should be at least min_batch_size
        let target = controller.eviction_target(1);
        assert!(target >= controller.config.min_batch_size);

        // Target should include hysteresis factor
        let target = controller.eviction_target(100);
        assert!(target > 100);
    }

    #[test]
    fn test_pressure_levels() {
        assert_eq!(PressureLevel::from_utilization(0.3), PressureLevel::None);
        assert_eq!(PressureLevel::from_utilization(0.6), PressureLevel::Low);
        assert_eq!(PressureLevel::from_utilization(0.75), PressureLevel::Medium);
        assert_eq!(PressureLevel::from_utilization(0.90), PressureLevel::High);
        assert_eq!(
            PressureLevel::from_utilization(0.98),
            PressureLevel::Critical
        );
    }

    #[test]
    fn test_eviction_cycle() {
        let controller = HysteresisController::new(HysteresisConfig::default());

        assert!(!controller.is_evicting());

        controller.start_eviction();
        assert!(controller.is_evicting());
        assert_eq!(controller.evicted_in_cycle(), 0);

        controller.record_eviction(10);
        assert_eq!(controller.evicted_in_cycle(), 10);

        controller.record_eviction(5);
        assert_eq!(controller.evicted_in_cycle(), 15);

        controller.end_eviction();
        assert!(!controller.is_evicting());
        assert_eq!(controller.state(), HysteresisState::Cooldown);
    }
}
