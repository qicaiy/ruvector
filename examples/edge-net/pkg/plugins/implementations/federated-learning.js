/**
 * Federated Learning Plugin
 *
 * Train ML models across nodes without sharing raw data.
 * Implements FedAvg with differential privacy.
 *
 * @module @ruvector/edge-net/plugins/federated-learning
 */

import { EventEmitter } from 'events';
import { randomBytes } from 'crypto';

export class FederatedLearningPlugin extends EventEmitter {
    constructor(config = {}) {
        super();

        this.config = {
            aggregationStrategy: config.aggregationStrategy || 'fedavg',
            localEpochs: config.localEpochs || 5,
            differentialPrivacy: config.differentialPrivacy ?? true,
            noiseMultiplier: config.noiseMultiplier || 1.0,
            minParticipants: config.minParticipants || 3,
            roundTimeout: config.roundTimeout || 60000,
        };

        this.rounds = new Map();         // roundId -> RoundState
        this.localModels = new Map();    // modelId -> weights
        this.globalModels = new Map();   // modelId -> aggregated weights
    }

    /**
     * Start a new training round
     */
    startRound(modelId, globalWeights) {
        const roundId = `round-${Date.now()}-${randomBytes(4).toString('hex')}`;

        const round = {
            id: roundId,
            modelId,
            globalWeights,
            participants: new Map(),
            status: 'collecting',
            startedAt: Date.now(),
        };

        this.rounds.set(roundId, round);
        this.emit('round:started', { roundId, modelId });

        // Set timeout
        setTimeout(() => {
            if (round.status === 'collecting') {
                this._aggregateRound(roundId);
            }
        }, this.config.roundTimeout);

        return roundId;
    }

    /**
     * Train locally and submit update
     */
    async trainLocal(roundId, localData, options = {}) {
        const round = this.rounds.get(roundId);
        if (!round) {
            throw new Error(`Round not found: ${roundId}`);
        }

        // Simulate local training
        const localUpdate = await this._performLocalTraining(
            round.globalWeights,
            localData,
            options.epochs || this.config.localEpochs
        );

        // Add differential privacy noise if enabled
        if (this.config.differentialPrivacy) {
            this._addDifferentialPrivacy(localUpdate);
        }

        // Submit update
        const participantId = options.participantId || randomBytes(8).toString('hex');
        round.participants.set(participantId, {
            update: localUpdate,
            dataSize: localData.length,
            submittedAt: Date.now(),
        });

        this.emit('update:submitted', { roundId, participantId });

        // Check if we have enough participants
        if (round.participants.size >= this.config.minParticipants) {
            this._aggregateRound(roundId);
        }

        return { participantId, updateSize: localUpdate.length };
    }

    /**
     * Perform local training (simulated)
     */
    async _performLocalTraining(globalWeights, localData, epochs) {
        // In production: Use ONNX Runtime or TensorFlow.js
        // For demo: Simulate gradient descent
        const weights = globalWeights ? [...globalWeights] : Array(10).fill(0);

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (const sample of localData) {
                // Simplified SGD update
                for (let i = 0; i < weights.length; i++) {
                    const gradient = (sample.features?.[i] || Math.random()) * 0.01;
                    weights[i] -= gradient;
                }
            }
        }

        return weights;
    }

    /**
     * Add differential privacy noise
     */
    _addDifferentialPrivacy(weights) {
        const sigma = this.config.noiseMultiplier;
        for (let i = 0; i < weights.length; i++) {
            // Gaussian noise
            const u1 = Math.random();
            const u2 = Math.random();
            const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            weights[i] += noise * sigma;
        }
    }

    /**
     * Aggregate updates using FedAvg
     */
    _aggregateRound(roundId) {
        const round = this.rounds.get(roundId);
        if (!round || round.status !== 'collecting') {
            return;
        }

        round.status = 'aggregating';

        const updates = Array.from(round.participants.values());
        if (updates.length === 0) {
            round.status = 'failed';
            this.emit('round:failed', { roundId, reason: 'No updates' });
            return;
        }

        let aggregatedWeights;

        switch (this.config.aggregationStrategy) {
            case 'fedavg':
                aggregatedWeights = this._fedAvg(updates);
                break;
            case 'fedprox':
                aggregatedWeights = this._fedProx(updates, round.globalWeights);
                break;
            default:
                aggregatedWeights = this._fedAvg(updates);
        }

        // Store aggregated model
        this.globalModels.set(round.modelId, aggregatedWeights);
        round.status = 'completed';
        round.aggregatedWeights = aggregatedWeights;
        round.completedAt = Date.now();

        this.emit('round:completed', {
            roundId,
            modelId: round.modelId,
            participants: round.participants.size,
            duration: round.completedAt - round.startedAt,
        });

        return aggregatedWeights;
    }

    /**
     * FedAvg aggregation
     */
    _fedAvg(updates) {
        if (updates.length === 0) return null;

        const totalSamples = updates.reduce((sum, u) => sum + u.dataSize, 0);
        const numWeights = updates[0].update.length;
        const aggregated = Array(numWeights).fill(0);

        for (const { update, dataSize } of updates) {
            const weight = dataSize / totalSamples;
            for (let i = 0; i < numWeights; i++) {
                aggregated[i] += update[i] * weight;
            }
        }

        return aggregated;
    }

    /**
     * FedProx aggregation (with proximal term)
     */
    _fedProx(updates, globalWeights) {
        const fedAvgResult = this._fedAvg(updates);
        if (!globalWeights) return fedAvgResult;

        // Add proximal regularization
        const mu = 0.01; // Proximal strength
        for (let i = 0; i < fedAvgResult.length; i++) {
            fedAvgResult[i] = (1 - mu) * fedAvgResult[i] + mu * globalWeights[i];
        }

        return fedAvgResult;
    }

    /**
     * Get current global model
     */
    getGlobalModel(modelId) {
        return this.globalModels.get(modelId);
    }

    /**
     * Get round status
     */
    getRoundStatus(roundId) {
        const round = this.rounds.get(roundId);
        if (!round) return null;

        return {
            id: round.id,
            modelId: round.modelId,
            status: round.status,
            participants: round.participants.size,
            startedAt: round.startedAt,
            completedAt: round.completedAt,
        };
    }

    getStats() {
        return {
            totalRounds: this.rounds.size,
            globalModels: this.globalModels.size,
            config: this.config,
        };
    }
}

export default FederatedLearningPlugin;
