/**
 * Swarm Intelligence Plugin
 *
 * Distributed optimization using bio-inspired algorithms:
 * - Particle Swarm Optimization (PSO)
 * - Ant Colony Optimization (ACO)
 * - Genetic Algorithm (GA)
 * - Differential Evolution (DE)
 *
 * @module @ruvector/edge-net/plugins/swarm-intelligence
 */

import { EventEmitter } from 'events';

export class SwarmIntelligencePlugin extends EventEmitter {
    constructor(config = {}) {
        super();

        this.config = {
            algorithm: config.algorithm || 'pso',
            populationSize: config.populationSize || 50,
            iterations: config.iterations || 100,
            dimensions: config.dimensions || 10,
        };

        this.swarms = new Map(); // swarmId -> SwarmState
    }

    /**
     * Create optimization swarm
     */
    createSwarm(swarmId, options = {}) {
        const swarm = {
            id: swarmId,
            algorithm: options.algorithm || this.config.algorithm,
            dimensions: options.dimensions || this.config.dimensions,
            bounds: options.bounds || { min: -10, max: 10 },
            fitnessFunction: options.fitnessFunction || this._defaultFitness,
            particles: [],
            globalBest: null,
            globalBestFitness: Infinity,
            iteration: 0,
            status: 'initialized',
        };

        // Initialize particles
        for (let i = 0; i < this.config.populationSize; i++) {
            swarm.particles.push(this._createParticle(swarm));
        }

        this.swarms.set(swarmId, swarm);
        this.emit('swarm:created', { swarmId, algorithm: swarm.algorithm });

        return swarm;
    }

    /**
     * Create single particle
     */
    _createParticle(swarm) {
        const position = Array(swarm.dimensions).fill(0).map(() =>
            swarm.bounds.min + Math.random() * (swarm.bounds.max - swarm.bounds.min)
        );

        return {
            position,
            velocity: Array(swarm.dimensions).fill(0).map(() => (Math.random() - 0.5) * 2),
            bestPosition: [...position],
            bestFitness: Infinity,
            fitness: Infinity,
        };
    }

    /**
     * Default fitness function (minimize sum of squares)
     */
    _defaultFitness(position) {
        return position.reduce((sum, x) => sum + x * x, 0);
    }

    /**
     * Run optimization step
     */
    step(swarmId) {
        const swarm = this.swarms.get(swarmId);
        if (!swarm) throw new Error(`Swarm not found: ${swarmId}`);

        swarm.status = 'running';
        swarm.iteration++;

        switch (swarm.algorithm) {
            case 'pso':
                this._psoStep(swarm);
                break;
            case 'ga':
                this._gaStep(swarm);
                break;
            case 'de':
                this._deStep(swarm);
                break;
            case 'aco':
                this._acoStep(swarm);
                break;
            default:
                this._psoStep(swarm);
        }

        this.emit('swarm:step', {
            swarmId,
            iteration: swarm.iteration,
            bestFitness: swarm.globalBestFitness,
        });

        return {
            iteration: swarm.iteration,
            bestFitness: swarm.globalBestFitness,
            bestPosition: swarm.globalBest,
        };
    }

    /**
     * Particle Swarm Optimization step
     */
    _psoStep(swarm) {
        const w = 0.7;   // Inertia weight
        const c1 = 1.5;  // Cognitive coefficient
        const c2 = 1.5;  // Social coefficient

        for (const particle of swarm.particles) {
            // Update velocity
            for (let d = 0; d < swarm.dimensions; d++) {
                const r1 = Math.random();
                const r2 = Math.random();

                particle.velocity[d] =
                    w * particle.velocity[d] +
                    c1 * r1 * (particle.bestPosition[d] - particle.position[d]) +
                    c2 * r2 * ((swarm.globalBest?.[d] || 0) - particle.position[d]);
            }

            // Update position
            for (let d = 0; d < swarm.dimensions; d++) {
                particle.position[d] += particle.velocity[d];
                // Clamp to bounds
                particle.position[d] = Math.max(swarm.bounds.min,
                    Math.min(swarm.bounds.max, particle.position[d]));
            }

            // Evaluate fitness
            particle.fitness = swarm.fitnessFunction(particle.position);

            // Update personal best
            if (particle.fitness < particle.bestFitness) {
                particle.bestFitness = particle.fitness;
                particle.bestPosition = [...particle.position];
            }

            // Update global best
            if (particle.fitness < swarm.globalBestFitness) {
                swarm.globalBestFitness = particle.fitness;
                swarm.globalBest = [...particle.position];
            }
        }
    }

    /**
     * Genetic Algorithm step
     */
    _gaStep(swarm) {
        const mutationRate = 0.1;
        const crossoverRate = 0.8;

        // Evaluate all
        for (const particle of swarm.particles) {
            particle.fitness = swarm.fitnessFunction(particle.position);
            if (particle.fitness < swarm.globalBestFitness) {
                swarm.globalBestFitness = particle.fitness;
                swarm.globalBest = [...particle.position];
            }
        }

        // Selection (tournament)
        const selected = [];
        for (let i = 0; i < swarm.particles.length; i++) {
            const a = swarm.particles[Math.floor(Math.random() * swarm.particles.length)];
            const b = swarm.particles[Math.floor(Math.random() * swarm.particles.length)];
            selected.push(a.fitness < b.fitness ? a : b);
        }

        // Crossover and mutation
        for (let i = 0; i < swarm.particles.length; i += 2) {
            const p1 = selected[i];
            const p2 = selected[i + 1] || selected[0];

            if (Math.random() < crossoverRate && swarm.particles[i + 1]) {
                const crossPoint = Math.floor(Math.random() * swarm.dimensions);
                for (let d = crossPoint; d < swarm.dimensions; d++) {
                    const temp = swarm.particles[i].position[d];
                    swarm.particles[i].position[d] = p2.position[d];
                    swarm.particles[i + 1].position[d] = temp;
                }
            }

            // Mutation
            for (const particle of [swarm.particles[i], swarm.particles[i + 1]]) {
                if (!particle) continue;
                for (let d = 0; d < swarm.dimensions; d++) {
                    if (Math.random() < mutationRate) {
                        particle.position[d] += (Math.random() - 0.5) *
                            (swarm.bounds.max - swarm.bounds.min) * 0.1;
                        particle.position[d] = Math.max(swarm.bounds.min,
                            Math.min(swarm.bounds.max, particle.position[d]));
                    }
                }
            }
        }
    }

    /**
     * Differential Evolution step
     */
    _deStep(swarm) {
        const F = 0.8;  // Mutation factor
        const CR = 0.9; // Crossover rate

        const newPositions = [];

        for (let i = 0; i < swarm.particles.length; i++) {
            // Select 3 random distinct particles
            const indices = [];
            while (indices.length < 3) {
                const idx = Math.floor(Math.random() * swarm.particles.length);
                if (idx !== i && !indices.includes(idx)) indices.push(idx);
            }

            const [a, b, c] = indices.map(idx => swarm.particles[idx]);

            // Create mutant
            const mutant = a.position.map((v, d) =>
                v + F * (b.position[d] - c.position[d])
            );

            // Crossover
            const trial = swarm.particles[i].position.map((v, d) =>
                Math.random() < CR ? mutant[d] : v
            );

            // Clamp
            trial.forEach((v, d) => {
                trial[d] = Math.max(swarm.bounds.min, Math.min(swarm.bounds.max, v));
            });

            const trialFitness = swarm.fitnessFunction(trial);

            if (trialFitness < swarm.particles[i].fitness) {
                newPositions.push({ idx: i, position: trial, fitness: trialFitness });
            }
        }

        // Apply updates
        for (const { idx, position, fitness } of newPositions) {
            swarm.particles[idx].position = position;
            swarm.particles[idx].fitness = fitness;

            if (fitness < swarm.globalBestFitness) {
                swarm.globalBestFitness = fitness;
                swarm.globalBest = [...position];
            }
        }
    }

    /**
     * Ant Colony Optimization step (simplified for continuous optimization)
     */
    _acoStep(swarm) {
        const evaporationRate = 0.1;
        const pheromoneDeposit = 1.0;

        // Use pheromone as bias toward best solutions
        if (!swarm.pheromone) {
            swarm.pheromone = Array(swarm.dimensions).fill(0);
        }

        for (const particle of swarm.particles) {
            // Move influenced by pheromone
            for (let d = 0; d < swarm.dimensions; d++) {
                const bias = swarm.pheromone[d] * 0.1;
                particle.position[d] += (Math.random() - 0.5) * 2 + bias;
                particle.position[d] = Math.max(swarm.bounds.min,
                    Math.min(swarm.bounds.max, particle.position[d]));
            }

            particle.fitness = swarm.fitnessFunction(particle.position);

            if (particle.fitness < swarm.globalBestFitness) {
                swarm.globalBestFitness = particle.fitness;
                swarm.globalBest = [...particle.position];
            }
        }

        // Update pheromone
        for (let d = 0; d < swarm.dimensions; d++) {
            swarm.pheromone[d] *= (1 - evaporationRate);
            if (swarm.globalBest) {
                swarm.pheromone[d] += pheromoneDeposit / (1 + Math.abs(swarm.globalBest[d]));
            }
        }
    }

    /**
     * Run full optimization
     */
    async optimize(swarmId, options = {}) {
        const maxIterations = options.iterations || this.config.iterations;
        const targetFitness = options.targetFitness ?? -Infinity;

        const swarm = this.swarms.get(swarmId);
        if (!swarm) throw new Error(`Swarm not found: ${swarmId}`);

        while (swarm.iteration < maxIterations && swarm.globalBestFitness > targetFitness) {
            this.step(swarmId);

            // Yield for async operation
            if (swarm.iteration % 10 === 0) {
                await new Promise(resolve => setImmediate(resolve));
            }
        }

        swarm.status = 'completed';

        this.emit('swarm:completed', {
            swarmId,
            iterations: swarm.iteration,
            bestFitness: swarm.globalBestFitness,
            bestPosition: swarm.globalBest,
        });

        return {
            iterations: swarm.iteration,
            bestFitness: swarm.globalBestFitness,
            bestPosition: swarm.globalBest,
        };
    }

    /**
     * Distribute optimization across network nodes
     */
    async distributeOptimization(swarmId, nodes, network) {
        const swarm = this.swarms.get(swarmId);
        if (!swarm) throw new Error(`Swarm not found: ${swarmId}`);

        // Split particles among nodes
        const particlesPerNode = Math.ceil(swarm.particles.length / nodes.length);
        const tasks = [];

        for (let i = 0; i < nodes.length; i++) {
            const startIdx = i * particlesPerNode;
            const endIdx = Math.min(startIdx + particlesPerNode, swarm.particles.length);
            const nodeParticles = swarm.particles.slice(startIdx, endIdx);

            tasks.push({
                nodeId: nodes[i],
                particles: nodeParticles.map(p => ({ position: p.position })),
                globalBest: swarm.globalBest,
                iteration: swarm.iteration,
            });
        }

        this.emit('distributed', { swarmId, nodeCount: nodes.length });

        return tasks;
    }

    getSwarm(swarmId) {
        return this.swarms.get(swarmId);
    }

    getStats() {
        return {
            activeSwarms: this.swarms.size,
            config: this.config,
        };
    }
}

export default SwarmIntelligencePlugin;
