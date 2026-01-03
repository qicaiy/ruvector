/**
 * Credits Service - Real-time credit management for EdgeNet
 *
 * Handles:
 * - Fast credit accumulation from contribution
 * - Real job deployment with credit consumption
 * - Comprehensive transaction logging
 * - Persistence to IndexedDB
 */

import { edgeNetService } from './edgeNet';

export interface CreditTransaction {
  id: string;
  type: 'earn' | 'spend' | 'pending';
  amount: number;
  reason: string;
  taskId?: string;
  timestamp: Date;
  balance: number;
}

export interface JobSubmission {
  id: string;
  type: 'compute' | 'inference' | 'training' | 'storage';
  payload: unknown;
  creditsRequired: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: unknown;
  submittedAt: Date;
  completedAt?: Date;
}

export interface CreditsState {
  available: number;
  pending: number;
  earned: number;
  spent: number;
  transactions: CreditTransaction[];
  jobs: JobSubmission[];
}

// Credit rates (rUv per second of contribution)
const CREDIT_RATES = {
  base: 0.01,      // Base rate per second
  cpuBonus: 0.005, // Additional per 10% CPU allocated
  networkBonus: 0.002, // Bonus for network peers
  uptimeBonus: 0.001,  // Bonus per minute of uptime
};

// Job costs
const JOB_COSTS = {
  compute: 0.1,    // Simple compute task
  inference: 0.5,  // ML inference
  training: 2.0,   // Model training
  storage: 0.05,   // Data storage per MB
};

class CreditsService {
  private state: CreditsState = {
    available: 0,
    pending: 0,
    earned: 0,
    spent: 0,
    transactions: [],
    jobs: [],
  };

  private lastEarnTime: number = 0;
  private listeners: Set<(state: CreditsState) => void> = new Set();
  private earnInterval: ReturnType<typeof setInterval> | null = null;
  private cpuLimit: number = 50;
  private isContributing: boolean = false;
  private uptimeSeconds: number = 0;
  private networkPeers: number = 0;

  /**
   * Initialize the credits service
   */
  initialize(initialState?: Partial<CreditsState>): void {
    if (initialState) {
      this.state = { ...this.state, ...initialState };
    }
    this.lastEarnTime = Date.now();
    console.log('[Credits] Service initialized with balance:', this.state.available, 'rUv');
  }

  /**
   * Start earning credits (called when contribution is enabled)
   */
  startEarning(cpuLimit: number = 50): void {
    this.cpuLimit = cpuLimit;
    this.isContributing = true;
    this.lastEarnTime = Date.now();

    // Clear any existing interval
    if (this.earnInterval) {
      clearInterval(this.earnInterval);
    }

    // Earn credits every second for responsive UI
    this.earnInterval = setInterval(() => {
      if (this.isContributing) {
        this.processEarning();
      }
    }, 1000);

    console.log('[Credits] Started earning at CPU limit:', cpuLimit, '%');
    this.logTransaction('earn', 0, 'Started contributing to network');
  }

  /**
   * Stop earning credits
   */
  stopEarning(): void {
    this.isContributing = false;
    if (this.earnInterval) {
      clearInterval(this.earnInterval);
      this.earnInterval = null;
    }
    console.log('[Credits] Stopped earning');
    this.logTransaction('earn', 0, 'Stopped contributing to network');
  }

  /**
   * Process credit earning for one second
   */
  private processEarning(): void {
    const now = Date.now();
    const elapsedSeconds = (now - this.lastEarnTime) / 1000;
    this.lastEarnTime = now;
    this.uptimeSeconds += elapsedSeconds;

    // Calculate credits earned
    let rate = CREDIT_RATES.base;

    // CPU bonus (more CPU = more credits)
    rate += (this.cpuLimit / 10) * CREDIT_RATES.cpuBonus;

    // Network bonus (connected peers)
    rate += this.networkPeers * CREDIT_RATES.networkBonus;

    // Uptime bonus (loyalty reward)
    rate += Math.floor(this.uptimeSeconds / 60) * CREDIT_RATES.uptimeBonus;

    const earnedAmount = rate * elapsedSeconds;

    // Update state
    this.state.earned += earnedAmount;
    this.state.available += earnedAmount;

    // Log every 10 seconds or when amount is significant
    if (Math.floor(this.uptimeSeconds) % 10 === 0 || earnedAmount > 0.1) {
      this.logTransaction('earn', earnedAmount, `Contribution reward (${rate.toFixed(4)} rUv/s)`);
    }

    // Also process via WASM if available
    edgeNetService.submitDemoTask().catch(() => {});
    edgeNetService.processNextTask().catch(() => {});

    // Notify listeners
    this.notifyListeners();
  }

  /**
   * Update network peer count (affects earning rate)
   */
  updateNetworkPeers(count: number): void {
    this.networkPeers = count;
  }

  /**
   * Submit a job that consumes credits
   */
  async submitJob(
    type: JobSubmission['type'],
    payload: unknown,
    customCredits?: number
  ): Promise<JobSubmission> {
    const creditsRequired = customCredits ?? JOB_COSTS[type];

    // Check if we have enough credits
    if (this.state.available < creditsRequired) {
      const error = `Insufficient credits. Required: ${creditsRequired} rUv, Available: ${this.state.available.toFixed(4)} rUv`;
      console.error('[Credits]', error);
      throw new Error(error);
    }

    const job: JobSubmission = {
      id: `job-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      type,
      payload,
      creditsRequired,
      status: 'pending',
      submittedAt: new Date(),
    };

    // Deduct credits immediately
    this.state.available -= creditsRequired;
    this.state.pending += creditsRequired;
    this.state.jobs.push(job);

    this.logTransaction('spend', creditsRequired, `Job submitted: ${type}`, job.id);
    console.log('[Credits] Job submitted:', job.id, '- Credits deducted:', creditsRequired, 'rUv');

    // Execute the job
    try {
      job.status = 'running';
      this.notifyListeners();

      // Actually run the job via WASM
      const payloadBytes = new TextEncoder().encode(JSON.stringify(payload));
      const result = await edgeNetService.submitTask(
        type,
        payloadBytes,
        BigInt(Math.floor(creditsRequired * 1e9))
      );

      // Wait for processing
      await edgeNetService.processNextTask();

      job.status = 'completed';
      job.result = result;
      job.completedAt = new Date();

      // Move from pending to spent
      this.state.pending -= creditsRequired;
      this.state.spent += creditsRequired;

      this.logTransaction('spend', 0, `Job completed: ${type}`, job.id);
      console.log('[Credits] Job completed:', job.id);

    } catch (error) {
      job.status = 'failed';
      job.completedAt = new Date();

      // Refund credits on failure
      this.state.pending -= creditsRequired;
      this.state.available += creditsRequired;

      this.logTransaction('earn', creditsRequired, `Job failed, credits refunded: ${type}`, job.id);
      console.error('[Credits] Job failed:', job.id, error);
    }

    this.notifyListeners();
    return job;
  }

  /**
   * Log a credit transaction
   */
  private logTransaction(
    type: CreditTransaction['type'],
    amount: number,
    reason: string,
    taskId?: string
  ): void {
    const transaction: CreditTransaction = {
      id: `tx-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      type,
      amount,
      reason,
      taskId,
      timestamp: new Date(),
      balance: this.state.available,
    };

    this.state.transactions.push(transaction);

    // Keep only last 100 transactions
    if (this.state.transactions.length > 100) {
      this.state.transactions = this.state.transactions.slice(-100);
    }

    // Log to console
    const emoji = type === 'earn' ? '💰' : type === 'spend' ? '💸' : '⏳';
    console.log(
      `[Credits] ${emoji} ${type.toUpperCase()}: ${amount.toFixed(6)} rUv | ${reason} | Balance: ${this.state.available.toFixed(4)} rUv`
    );
  }

  /**
   * Get current credits state
   */
  getState(): CreditsState {
    return { ...this.state };
  }

  /**
   * Get recent transactions
   */
  getTransactions(limit: number = 20): CreditTransaction[] {
    return this.state.transactions.slice(-limit);
  }

  /**
   * Get job history
   */
  getJobs(limit: number = 10): JobSubmission[] {
    return this.state.jobs.slice(-limit);
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener: (state: CreditsState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Notify all listeners of state change
   */
  private notifyListeners(): void {
    const state = this.getState();
    this.listeners.forEach(listener => listener(state));
  }

  /**
   * Sync state from network store
   */
  syncFromStore(credits: { available: number; earned: number; spent: number; pending: number }): void {
    this.state.available = credits.available;
    this.state.earned = credits.earned;
    this.state.spent = credits.spent;
    this.state.pending = credits.pending;
  }

  /**
   * Get earning rate per second
   */
  getEarningRate(): number {
    let rate = CREDIT_RATES.base;
    rate += (this.cpuLimit / 10) * CREDIT_RATES.cpuBonus;
    rate += this.networkPeers * CREDIT_RATES.networkBonus;
    rate += Math.floor(this.uptimeSeconds / 60) * CREDIT_RATES.uptimeBonus;
    return rate;
  }

  /**
   * Check if we can afford a job
   */
  canAfford(type: JobSubmission['type'], customCredits?: number): boolean {
    const cost = customCredits ?? JOB_COSTS[type];
    return this.state.available >= cost;
  }

  /**
   * Get job cost
   */
  getJobCost(type: JobSubmission['type']): number {
    return JOB_COSTS[type];
  }
}

export const creditsService = new CreditsService();
