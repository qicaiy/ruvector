/**
 * Reputation Staking Plugin
 *
 * Stake credits as collateral for good behavior.
 * Slashing mechanism for misbehavior detection.
 *
 * @module @ruvector/edge-net/plugins/reputation-staking
 */

import { EventEmitter } from 'events';

export class ReputationStakingPlugin extends EventEmitter {
    constructor(config = {}) {
        super();

        this.config = {
            minStake: config.minStake || 10,
            slashRate: config.slashRate || 0.1, // 10% slash
            unbondingPeriod: config.unbondingPeriod || 604800000, // 7 days
            maxSlashPerPeriod: config.maxSlashPerPeriod || 0.5, // Max 50% slash per period
        };

        // Staking state
        this.stakes = new Map();          // nodeId -> { staked, reputation, unbonding }
        this.slashHistory = new Map();    // nodeId -> [{ reason, amount, timestamp }]
        this.unbondingQueue = [];         // [{ nodeId, amount, availableAt }]
    }

    /**
     * Stake credits
     */
    stake(nodeId, amount, creditSystem) {
        if (amount < this.config.minStake) {
            throw new Error(`Minimum stake is ${this.config.minStake}`);
        }

        // Check balance
        const balance = creditSystem.getBalance(nodeId);
        if (balance < amount) {
            throw new Error(`Insufficient balance: ${balance} < ${amount}`);
        }

        // Lock credits
        creditSystem.spendCredits(nodeId, amount, `stake-${Date.now()}`);

        // Create or update stake
        let stake = this.stakes.get(nodeId);
        if (!stake) {
            stake = {
                staked: 0,
                reputation: 100, // Start at 100
                unbonding: 0,
                lastActivity: Date.now(),
                successfulTasks: 0,
                failedTasks: 0,
            };
            this.stakes.set(nodeId, stake);
        }

        stake.staked += amount;
        stake.lastActivity = Date.now();

        this.emit('staked', { nodeId, amount, totalStaked: stake.staked });

        return stake;
    }

    /**
     * Request unstaking (enters unbonding period)
     */
    unstake(nodeId, amount) {
        const stake = this.stakes.get(nodeId);
        if (!stake) {
            throw new Error(`No stake found for: ${nodeId}`);
        }

        if (amount > stake.staked) {
            throw new Error(`Cannot unstake more than staked: ${stake.staked}`);
        }

        stake.staked -= amount;
        stake.unbonding += amount;

        const availableAt = Date.now() + this.config.unbondingPeriod;
        this.unbondingQueue.push({ nodeId, amount, availableAt });

        this.emit('unstaking', { nodeId, amount, availableAt });

        return { unbonding: stake.unbonding, availableAt };
    }

    /**
     * Claim unbonded stake
     */
    claim(nodeId, creditSystem) {
        const now = Date.now();
        const stake = this.stakes.get(nodeId);
        if (!stake) {
            throw new Error(`No stake found for: ${nodeId}`);
        }

        let claimed = 0;
        this.unbondingQueue = this.unbondingQueue.filter(item => {
            if (item.nodeId === nodeId && item.availableAt <= now) {
                claimed += item.amount;
                stake.unbonding -= item.amount;
                return false;
            }
            return true;
        });

        if (claimed > 0) {
            creditSystem.earnCredits(nodeId, claimed, `unstake-claim-${Date.now()}`);
            this.emit('claimed', { nodeId, amount: claimed });
        }

        return { claimed };
    }

    /**
     * Slash stake for misbehavior
     */
    slash(nodeId, reason, severity = 1.0) {
        const stake = this.stakes.get(nodeId);
        if (!stake || stake.staked === 0) {
            return { slashed: 0, reason: 'No stake to slash' };
        }

        // Calculate slash amount
        const slashAmount = Math.min(
            stake.staked * this.config.slashRate * severity,
            stake.staked * this.config.maxSlashPerPeriod
        );

        stake.staked -= slashAmount;
        stake.reputation = Math.max(0, stake.reputation - 10 * severity);
        stake.failedTasks++;

        // Record slash history
        if (!this.slashHistory.has(nodeId)) {
            this.slashHistory.set(nodeId, []);
        }
        this.slashHistory.get(nodeId).push({
            reason,
            amount: slashAmount,
            severity,
            timestamp: Date.now(),
        });

        this.emit('slashed', { nodeId, amount: slashAmount, reason, newReputation: stake.reputation });

        return { slashed: slashAmount, newStake: stake.staked, newReputation: stake.reputation };
    }

    /**
     * Record successful task (increases reputation)
     */
    recordSuccess(nodeId) {
        const stake = this.stakes.get(nodeId);
        if (!stake) return;

        stake.successfulTasks++;
        stake.reputation = Math.min(100, stake.reputation + 1);
        stake.lastActivity = Date.now();

        this.emit('success', { nodeId, reputation: stake.reputation });
    }

    /**
     * Record failed task (may trigger slash)
     */
    recordFailure(nodeId, reason) {
        const stake = this.stakes.get(nodeId);
        if (!stake) return;

        stake.failedTasks++;
        stake.lastActivity = Date.now();

        // Calculate failure rate
        const totalTasks = stake.successfulTasks + stake.failedTasks;
        const failureRate = stake.failedTasks / totalTasks;

        // Slash if failure rate too high
        if (failureRate > 0.3 && totalTasks >= 10) {
            this.slash(nodeId, reason, failureRate);
        } else {
            // Just reduce reputation
            stake.reputation = Math.max(0, stake.reputation - 5);
        }
    }

    /**
     * Get stake info
     */
    getStake(nodeId) {
        return this.stakes.get(nodeId);
    }

    /**
     * Get reputation score
     */
    getReputation(nodeId) {
        const stake = this.stakes.get(nodeId);
        return stake?.reputation ?? 0;
    }

    /**
     * Get leaderboard
     */
    getLeaderboard(limit = 10) {
        return Array.from(this.stakes.entries())
            .map(([nodeId, stake]) => ({
                nodeId,
                staked: stake.staked,
                reputation: stake.reputation,
                successRate: stake.successfulTasks / (stake.successfulTasks + stake.failedTasks || 1),
            }))
            .sort((a, b) => b.reputation - a.reputation || b.staked - a.staked)
            .slice(0, limit);
    }

    /**
     * Check if node is eligible for tasks
     */
    isEligible(nodeId, minReputation = 50, minStake = 0) {
        const stake = this.stakes.get(nodeId);
        if (!stake) return false;

        return stake.reputation >= minReputation && stake.staked >= minStake;
    }

    getStats() {
        const stakes = Array.from(this.stakes.values());
        return {
            totalStaked: stakes.reduce((sum, s) => sum + s.staked, 0),
            totalUnbonding: stakes.reduce((sum, s) => sum + s.unbonding, 0),
            stakerCount: this.stakes.size,
            averageReputation: stakes.reduce((sum, s) => sum + s.reputation, 0) / (stakes.length || 1),
        };
    }
}

export default ReputationStakingPlugin;
