#!/usr/bin/env node

/**
 * RLM Training Dataset Generator
 *
 * Generates synthetic RLM training examples for task decomposition,
 * answer synthesis, and agent routing. Integrates with ReasoningBank
 * patterns for realistic training data.
 *
 * Usage:
 *   node rlm-dataset.js [--output <file>] [--count <n>] [--quality <min>]
 *
 * Example:
 *   node rlm-dataset.js --output rlm-training.json --count 1000 --quality 0.7
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Import routing dataset for agent definitions
const { AGENT_TRAINING_DATA, getDatasetStats } = require('./routing-dataset');

// =============================================================================
// Configuration
// =============================================================================

const DEFAULT_CONFIG = {
  outputFile: 'rlm-training-dataset.json',
  exampleCount: 500,
  minQuality: 0.6,
  hardNegativeRatio: 0.3,
  maxDecompositionDepth: 5,
  embeddingDim: 768,
  seed: 42,
};

// =============================================================================
// Decomposition Strategy Definitions
// =============================================================================

const DECOMPOSITION_STRATEGIES = {
  sequential: {
    name: 'sequential',
    description: 'Steps executed in order, each depending on the previous',
    complexityWeight: 1.5,
  },
  parallel: {
    name: 'parallel',
    description: 'Independent sub-queries that can run concurrently',
    complexityWeight: 1.5,
  },
  hierarchical: {
    name: 'hierarchical',
    description: 'Tree structure with parent-child dependencies',
    complexityWeight: 2.0,
  },
  'dag-based': {
    name: 'dag-based',
    description: 'Complex DAG with arbitrary dependencies',
    complexityWeight: 3.0,
  },
  iterative: {
    name: 'iterative',
    description: 'Query -> result -> refined query cycles',
    complexityWeight: 2.5,
  },
  none: {
    name: 'none',
    description: 'Simple query needing no decomposition',
    complexityWeight: 1.0,
  },
};

// =============================================================================
// Complex Query Templates
// =============================================================================

const COMPLEX_QUERY_TEMPLATES = [
  // Multi-step development tasks
  {
    template: 'Build a {feature} for the {system} that includes {requirement1} and {requirement2}',
    decomposition: ['research', 'design', 'implement', 'test', 'document'],
    strategy: 'sequential',
    agents: ['researcher', 'architect', 'coder', 'tester', 'documenter'],
    quality: 0.9,
  },
  {
    template: 'Refactor the {component} to use {pattern} while maintaining backward compatibility',
    decomposition: ['analyze', 'design_refactor', 'implement_changes', 'write_tests', 'review'],
    strategy: 'sequential',
    agents: ['researcher', 'architect', 'coder', 'tester', 'reviewer'],
    quality: 0.85,
  },
  {
    template: 'Add {feature} support to the application with proper error handling and logging',
    decomposition: ['requirements', 'implementation', 'error_handling', 'logging', 'testing'],
    strategy: 'parallel',
    agents: ['researcher', 'coder', 'coder', 'coder', 'tester'],
    quality: 0.88,
  },
  {
    template: 'Investigate and fix the performance issue in {component} causing {symptom}',
    decomposition: ['investigate', 'identify_cause', 'design_fix', 'implement_fix', 'verify'],
    strategy: 'sequential',
    agents: ['researcher', 'debugger', 'architect', 'coder', 'tester'],
    quality: 0.82,
  },
  {
    template: 'Create a comprehensive API for {domain} with authentication, rate limiting, and documentation',
    decomposition: ['design_api', 'implement_endpoints', 'add_auth', 'add_rate_limiting', 'write_docs'],
    strategy: 'hierarchical',
    agents: ['architect', 'coder', 'security-architect', 'coder', 'api-docs'],
    quality: 0.9,
  },
  {
    template: 'Migrate the {system} from {old_tech} to {new_tech} with minimal downtime',
    decomposition: ['plan_migration', 'prepare_infrastructure', 'migrate_data', 'switch_traffic', 'cleanup'],
    strategy: 'sequential',
    agents: ['planner', 'devops', 'coder', 'devops', 'devops'],
    quality: 0.87,
  },
  {
    template: 'Implement a {algorithm} for {use_case} optimized for {constraint}',
    decomposition: ['research_approaches', 'design_algorithm', 'implement', 'optimize', 'benchmark'],
    strategy: 'sequential',
    agents: ['researcher', 'architect', 'coder', 'optimizer', 'tester'],
    quality: 0.85,
  },
  {
    template: 'Set up CI/CD pipeline for {project} with automated testing, security scanning, and deployment',
    decomposition: ['design_pipeline', 'setup_tests', 'add_security', 'configure_deployment', 'document'],
    strategy: 'dag-based',
    agents: ['devops', 'tester', 'security-architect', 'devops', 'documenter'],
    quality: 0.88,
  },
  {
    template: 'Review and improve the security of {component} following OWASP guidelines',
    decomposition: ['audit_current', 'identify_vulnerabilities', 'plan_fixes', 'implement_fixes', 'verify'],
    strategy: 'sequential',
    agents: ['security-architect', 'security-architect', 'architect', 'coder', 'security-architect'],
    quality: 0.9,
  },
  {
    template: 'Design and implement a {data_structure} optimized for {operation} operations',
    decomposition: ['research_options', 'design_structure', 'implement', 'write_tests', 'benchmark'],
    strategy: 'sequential',
    agents: ['researcher', 'architect', 'coder', 'tester', 'optimizer'],
    quality: 0.85,
  },
];

// Fill-in values for templates
const TEMPLATE_VALUES = {
  feature: ['dark mode', 'real-time sync', 'offline support', 'multi-language', 'analytics dashboard', 'notification system', 'file upload', 'export functionality'],
  system: ['web application', 'mobile app', 'backend service', 'CLI tool', 'browser extension', 'desktop app'],
  requirement1: ['user authentication', 'data validation', 'error handling', 'caching', 'pagination', 'search'],
  requirement2: ['logging', 'metrics', 'accessibility', 'responsive design', 'keyboard shortcuts', 'undo/redo'],
  component: ['authentication module', 'payment service', 'user profile', 'notification system', 'search engine', 'data pipeline'],
  pattern: ['async/await', 'event sourcing', 'CQRS', 'microservices', 'domain-driven design', 'hexagonal architecture'],
  symptom: ['slow response times', 'high memory usage', 'database connection leaks', 'timeout errors', 'intermittent failures'],
  domain: ['user management', 'inventory', 'payments', 'analytics', 'content management', 'scheduling'],
  old_tech: ['MongoDB', 'Express.js', 'JavaScript', 'REST API', 'monolith', 'SQL Server'],
  new_tech: ['PostgreSQL', 'NestJS', 'TypeScript', 'GraphQL', 'microservices', 'MongoDB'],
  algorithm: ['search algorithm', 'sorting algorithm', 'caching strategy', 'load balancing', 'rate limiting', 'data compression'],
  use_case: ['real-time search', 'batch processing', 'stream processing', 'recommendation engine', 'fraud detection'],
  constraint: ['memory efficiency', 'low latency', 'high throughput', 'horizontal scaling', 'minimal dependencies'],
  project: ['Node.js application', 'React frontend', 'Python service', 'Rust library', 'Go microservice'],
  data_structure: ['priority queue', 'LRU cache', 'trie', 'bloom filter', 'skip list', 'B-tree'],
  operation: ['insert', 'search', 'delete', 'range query', 'bulk update'],
};

// =============================================================================
// Hard Negative Pairs (Confusable Agent Combinations)
// =============================================================================

const HARD_NEGATIVE_PAIRS = [
  ['coder', 'debugger'],
  ['coder', 'refactorer'],
  ['researcher', 'reviewer'],
  ['tester', 'reviewer'],
  ['architect', 'planner'],
  ['documenter', 'api-docs'],
  ['optimizer', 'debugger'],
  ['devops', 'architect'],
  ['security-architect', 'reviewer'],
];

// =============================================================================
// Utility Functions
// =============================================================================

function generateUUID() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older Node versions
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function seededRandom(seed) {
  let state = seed;
  return function() {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function shuffle(array, rng) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function pickRandom(array, rng) {
  return array[Math.floor(rng() * array.length)];
}

function generateEmbedding(dim, rng) {
  const embedding = [];
  for (let i = 0; i < dim; i++) {
    embedding.push(rng() * 2 - 1);
  }
  // Normalize
  const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
  return embedding.map(v => v / norm);
}

function isHardNegative(agent1, agent2) {
  return HARD_NEGATIVE_PAIRS.some(
    ([a, b]) => (agent1 === a && agent2 === b) || (agent1 === b && agent2 === a)
  );
}

// =============================================================================
// Dataset Generation Functions
// =============================================================================

/**
 * Fill a template with random values
 */
function fillTemplate(template, rng) {
  let filled = template;
  for (const [key, values] of Object.entries(TEMPLATE_VALUES)) {
    const regex = new RegExp(`\\{${key}\\}`, 'g');
    filled = filled.replace(regex, () => pickRandom(values, rng));
  }
  return filled;
}

/**
 * Generate a sub-query from a decomposition step
 */
function generateSubQuery(id, step, agent, rng, dependencies = []) {
  const stepDescriptions = {
    research: 'Research best practices and options for',
    design: 'Design the architecture for',
    implement: 'Implement the functionality for',
    test: 'Write comprehensive tests for',
    document: 'Document the implementation of',
    analyze: 'Analyze the current state of',
    design_refactor: 'Design the refactoring approach for',
    implement_changes: 'Implement the refactored code for',
    write_tests: 'Write regression tests for',
    review: 'Review the code quality of',
    requirements: 'Gather and analyze requirements for',
    implementation: 'Build the core implementation of',
    error_handling: 'Add error handling to',
    logging: 'Implement logging for',
    testing: 'Create test suite for',
    investigate: 'Investigate the root cause of',
    identify_cause: 'Identify the specific cause of',
    design_fix: 'Design a fix for',
    implement_fix: 'Implement the fix for',
    verify: 'Verify the fix works for',
    design_api: 'Design the API structure for',
    implement_endpoints: 'Implement API endpoints for',
    add_auth: 'Add authentication to',
    add_rate_limiting: 'Implement rate limiting for',
    write_docs: 'Write API documentation for',
    plan_migration: 'Create migration plan for',
    prepare_infrastructure: 'Prepare infrastructure for',
    migrate_data: 'Migrate data for',
    switch_traffic: 'Switch traffic for',
    cleanup: 'Clean up old resources for',
    research_approaches: 'Research possible approaches for',
    design_algorithm: 'Design the algorithm for',
    optimize: 'Optimize performance of',
    benchmark: 'Benchmark performance of',
    design_pipeline: 'Design CI/CD pipeline for',
    setup_tests: 'Set up automated testing for',
    add_security: 'Add security scanning to',
    configure_deployment: 'Configure deployment for',
    audit_current: 'Audit current security of',
    identify_vulnerabilities: 'Identify vulnerabilities in',
    plan_fixes: 'Plan security fixes for',
    implement_fixes: 'Implement security fixes for',
    research_options: 'Research implementation options for',
    design_structure: 'Design data structure for',
  };

  const description = stepDescriptions[step] || `Process ${step} for`;

  return {
    id,
    query: `${description} the component`,
    expectedType: agent === 'coder' ? 'code' : agent === 'documenter' || agent === 'api-docs' ? 'documentation' : 'analysis',
    dependencies,
    recommendedAgent: agent,
    complexity: 0.3 + rng() * 0.5,
    context: null,
  };
}

/**
 * Generate sub-answers for sub-queries
 */
function generateSubAnswers(subQueries, rng, successRate = 0.9) {
  return subQueries.map(sq => ({
    subQueryId: sq.id,
    content: `Completed ${sq.query}`,
    confidence: 0.7 + rng() * 0.3,
    agent: sq.recommendedAgent,
    latencyMs: Math.floor(100 + rng() * 2000),
    quality: rng() < successRate ? 0.7 + rng() * 0.3 : 0.3 + rng() * 0.4,
    success: rng() < successRate,
    error: rng() >= successRate ? 'Partial completion' : null,
    reasoning: null,
  }));
}

/**
 * Generate dependencies based on strategy
 */
function generateDependencies(subQueries, strategy) {
  switch (strategy) {
    case 'sequential':
      return subQueries.map((sq, i) => ({
        ...sq,
        dependencies: i > 0 ? [i - 1] : [],
      }));

    case 'parallel':
      return subQueries.map(sq => ({
        ...sq,
        dependencies: [],
      }));

    case 'hierarchical':
      // First query is root, others depend on it or previous
      return subQueries.map((sq, i) => ({
        ...sq,
        dependencies: i === 0 ? [] : i < 3 ? [0] : [Math.floor(i / 2)],
      }));

    case 'dag-based':
      // Complex dependencies
      return subQueries.map((sq, i) => {
        const deps = [];
        if (i > 0) deps.push(i - 1);
        if (i > 2) deps.push(0);
        return { ...sq, dependencies: deps };
      });

    case 'iterative':
      return subQueries.map((sq, i) => ({
        ...sq,
        dependencies: i > 0 ? [i - 1] : [],
      }));

    default:
      return subQueries;
  }
}

/**
 * Generate a single RLM training example
 */
function generateExample(config, rng) {
  const template = pickRandom(COMPLEX_QUERY_TEMPLATES, rng);
  const query = fillTemplate(template.template, rng);

  // Generate sub-queries
  let subQueries = template.decomposition.map((step, i) =>
    generateSubQuery(i, step, template.agents[i], rng)
  );

  // Add dependencies based on strategy
  subQueries = generateDependencies(subQueries, template.strategy);

  const strategyDef = DECOMPOSITION_STRATEGIES[template.strategy];
  const totalComplexity = subQueries.reduce((sum, sq) => sum + sq.complexity, 0) * strategyDef.complexityWeight;

  // Generate decomposition
  const decomposition = {
    subQueries,
    strategy: template.strategy,
    rationale: `Using ${template.strategy} strategy for optimal execution`,
    totalComplexity,
    success: true,
    error: null,
  };

  // Generate sub-answers
  const subAnswers = generateSubAnswers(subQueries, rng, template.quality);

  // Calculate quality score
  const avgSubAnswerQuality = subAnswers.length > 0
    ? subAnswers.reduce((sum, a) => sum + a.quality, 0) / subAnswers.length
    : 0;

  const success = subAnswers.every(a => a.success);
  const qualityScore = success
    ? template.quality * (0.9 + rng() * 0.1)
    : template.quality * (0.5 + rng() * 0.3);

  // Generate trajectory metadata
  const trajectory = {
    sessionId: generateUUID(),
    userId: null,
    totalLatencyMs: subAnswers.reduce((sum, a) => sum + a.latencyMs, 0),
    retries: success ? 0 : Math.floor(rng() * 3),
    maxParallelism: template.strategy === 'parallel' ? subQueries.length : 1,
    modelsUsed: ['ruvltra-0.5b'],
    agentsInvoked: [...new Set(subQueries.map(sq => sq.recommendedAgent))],
    toolsUsed: [],
    attributes: {},
  };

  return {
    id: generateUUID(),
    query,
    queryEmbedding: generateEmbedding(config.embeddingDim, rng),
    decomposition,
    subAnswers,
    finalAnswer: `Successfully completed: ${query}`,
    finalEmbedding: generateEmbedding(config.embeddingDim, rng),
    qualityScore,
    trajectory,
    success,
    lessons: success ? [] : ['Encountered challenges during execution'],
    source: 'synthetic',
  };
}

/**
 * Generate contrastive pairs from examples
 */
function generateContrastivePairs(examples, config, rng) {
  const pairs = [];
  const agents = Object.keys(AGENT_TRAINING_DATA);

  for (const example of examples) {
    for (const subQuery of example.decomposition.subQueries) {
      if (!subQuery.recommendedAgent) continue;

      const positiveAgent = subQuery.recommendedAgent;

      for (const negativeAgent of agents) {
        if (negativeAgent === positiveAgent) continue;

        const isHard = isHardNegative(positiveAgent, negativeAgent);

        // Apply hard negative ratio
        const include = isHard
          ? rng() < config.hardNegativeRatio
          : rng() < (1 - config.hardNegativeRatio) / agents.length;

        if (include) {
          pairs.push({
            anchor: subQuery.query,
            anchorEmbedding: example.queryEmbedding,
            positiveAgent,
            negativeAgent,
            isHardNegative: isHard,
            quality: example.qualityScore,
            sourceId: example.id,
          });
        }
      }
    }
  }

  return pairs;
}

/**
 * Generate the complete RLM dataset
 */
function generateRlmDataset(config = {}) {
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  const rng = seededRandom(finalConfig.seed);

  console.log('\n============================================================');
  console.log('           RLM TRAINING DATASET GENERATOR');
  console.log('============================================================\n');

  console.log('Configuration:');
  console.log(`  Examples to generate: ${finalConfig.exampleCount}`);
  console.log(`  Minimum quality: ${finalConfig.minQuality}`);
  console.log(`  Hard negative ratio: ${finalConfig.hardNegativeRatio}`);
  console.log(`  Embedding dimension: ${finalConfig.embeddingDim}`);
  console.log(`  Random seed: ${finalConfig.seed}`);
  console.log('');

  // Generate examples
  console.log('Generating examples...');
  const examples = [];
  let attempts = 0;
  const maxAttempts = finalConfig.exampleCount * 2;

  while (examples.length < finalConfig.exampleCount && attempts < maxAttempts) {
    const example = generateExample(finalConfig, rng);
    if (example.qualityScore >= finalConfig.minQuality) {
      examples.push(example);
    }
    attempts++;
  }

  console.log(`  Generated ${examples.length} examples (${attempts} attempts)`);

  // Generate contrastive pairs
  console.log('\nGenerating contrastive pairs...');
  const contrastivePairs = generateContrastivePairs(examples, finalConfig, rng);
  console.log(`  Generated ${contrastivePairs.length} contrastive pairs`);

  // Calculate statistics
  const stats = {
    totalExamples: examples.length,
    successfulExamples: examples.filter(e => e.success).length,
    avgQuality: examples.reduce((sum, e) => sum + e.qualityScore, 0) / examples.length,
    avgDecompositionDepth: examples.reduce((sum, e) => sum + e.decomposition.subQueries.length, 0) / examples.length,
    strategyDistribution: {},
    agentDistribution: {},
    contrastivePairs: contrastivePairs.length,
    hardNegativePairs: contrastivePairs.filter(p => p.isHardNegative).length,
  };

  // Calculate strategy distribution
  for (const example of examples) {
    const strategy = example.decomposition.strategy;
    stats.strategyDistribution[strategy] = (stats.strategyDistribution[strategy] || 0) + 1;
  }

  // Calculate agent distribution
  for (const example of examples) {
    for (const sq of example.decomposition.subQueries) {
      const agent = sq.recommendedAgent;
      stats.agentDistribution[agent] = (stats.agentDistribution[agent] || 0) + 1;
    }
  }

  console.log('\n============================================================');
  console.log('                    DATASET STATISTICS');
  console.log('============================================================\n');
  console.log(`Total Examples:           ${stats.totalExamples}`);
  console.log(`Successful Examples:      ${stats.successfulExamples}`);
  console.log(`Average Quality:          ${(stats.avgQuality * 100).toFixed(1)}%`);
  console.log(`Avg Decomposition Depth:  ${stats.avgDecompositionDepth.toFixed(1)} steps`);
  console.log(`Contrastive Pairs:        ${stats.contrastivePairs}`);
  console.log(`Hard Negative Pairs:      ${stats.hardNegativePairs} (${((stats.hardNegativePairs / stats.contrastivePairs) * 100).toFixed(1)}%)`);

  console.log('\nStrategy Distribution:');
  for (const [strategy, count] of Object.entries(stats.strategyDistribution)) {
    console.log(`  ${strategy.padEnd(15)} ${count} (${((count / stats.totalExamples) * 100).toFixed(1)}%)`);
  }

  console.log('\nAgent Distribution:');
  const sortedAgents = Object.entries(stats.agentDistribution).sort((a, b) => b[1] - a[1]);
  for (const [agent, count] of sortedAgents) {
    console.log(`  ${agent.padEnd(20)} ${count}`);
  }

  return {
    examples,
    contrastivePairs,
    stats,
    config: finalConfig,
  };
}

/**
 * Export dataset to JSON file
 */
function exportDataset(dataset, outputFile) {
  const output = {
    metadata: {
      generatedAt: new Date().toISOString(),
      version: '1.0.0',
      config: dataset.config,
      stats: dataset.stats,
    },
    examples: dataset.examples,
    contrastivePairs: dataset.contrastivePairs,
  };

  fs.writeFileSync(outputFile, JSON.stringify(output, null, 2));
  console.log(`\nDataset exported to: ${outputFile}`);
  console.log(`File size: ${(fs.statSync(outputFile).size / 1024 / 1024).toFixed(2)} MB`);
}

// =============================================================================
// CLI Interface
// =============================================================================

function parseArgs(args) {
  const config = { ...DEFAULT_CONFIG };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--output' && args[i + 1]) {
      config.outputFile = args[++i];
    } else if (arg === '--count' && args[i + 1]) {
      config.exampleCount = parseInt(args[++i], 10);
    } else if (arg === '--quality' && args[i + 1]) {
      config.minQuality = parseFloat(args[++i]);
    } else if (arg === '--seed' && args[i + 1]) {
      config.seed = parseInt(args[++i], 10);
    } else if (arg === '--hard-ratio' && args[i + 1]) {
      config.hardNegativeRatio = parseFloat(args[++i]);
    } else if (arg === '--help' || arg === '-h') {
      console.log(`
RLM Training Dataset Generator

Usage: node rlm-dataset.js [options]

Options:
  --output <file>     Output file path (default: rlm-training-dataset.json)
  --count <n>         Number of examples to generate (default: 500)
  --quality <min>     Minimum quality threshold (default: 0.6)
  --seed <n>          Random seed (default: 42)
  --hard-ratio <r>    Hard negative pair ratio (default: 0.3)
  --help, -h          Show this help message

Example:
  node rlm-dataset.js --output my-dataset.json --count 1000 --quality 0.7
`);
      process.exit(0);
    }
  }

  return config;
}

// Main execution
if (require.main === module) {
  const config = parseArgs(process.argv.slice(2));
  const dataset = generateRlmDataset(config);
  exportDataset(dataset, config.outputFile);
}

// Module exports
module.exports = {
  generateRlmDataset,
  generateExample,
  generateContrastivePairs,
  exportDataset,
  COMPLEX_QUERY_TEMPLATES,
  DECOMPOSITION_STRATEGIES,
  TEMPLATE_VALUES,
  HARD_NEGATIVE_PAIRS,
  DEFAULT_CONFIG,
};
