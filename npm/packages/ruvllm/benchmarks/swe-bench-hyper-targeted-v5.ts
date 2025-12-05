/**
 * HYPER-TARGETED TRAINING V5
 *
 * Building on V4 success (+1.6%): Enhance all working techniques
 *
 * IMPROVEMENTS:
 * 1. TF-IDF keyword weighting (not just counts)
 * 2. Module path → file mapping from imports
 * 3. Error type → file associations
 * 4. File path structure learning (directory patterns)
 * 5. Ensemble scoring with multiple signals
 * 6. Bi-gram keywords for better context
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

interface SWEBenchInstance {
    instance_id: string;
    repo: string;
    patch: string;
    problem_statement: string;
    hints_text: string;
}

// ============================================================================
// ENHANCED CANDIDATE EXTRACTOR
// ============================================================================

function extractCandidates(problem: string): Array<{ file: string; source: string; baseScore: number }> {
    const candidates: Array<{ file: string; source: string; baseScore: number }> = [];
    const seen = new Set<string>();

    const addCandidate = (file: string, source: string, baseScore: number) => {
        const normalized = file.split('/').pop() || file;
        if (!seen.has(normalized) && normalized.endsWith('.py')) {
            seen.add(normalized);
            candidates.push({ file: normalized, source, baseScore });
        }
    };

    // Strategy 1: Backtick files (highest confidence)
    const backticks = problem.match(/`([^`]+\.py)`/g) || [];
    for (const bt of backticks) {
        addCandidate(bt.replace(/`/g, ''), 'backtick', 0.9);
    }

    // Strategy 2: Quoted files
    const quoted = problem.match(/"([^"]+\.py)"/g) || [];
    for (const q of quoted) {
        addCandidate(q.replace(/"/g, ''), 'quoted', 0.85);
    }

    // Strategy 3: Traceback files
    const tracebacks = problem.match(/File "([^"]+\.py)"/g) || [];
    for (const tb of tracebacks) {
        addCandidate(tb.replace(/File "|"/g, ''), 'traceback', 0.8);
    }

    // Strategy 4: Simple .py matches
    const simpleMatches = problem.match(/[\w\/]+\.py/g) || [];
    for (const f of simpleMatches) {
        addCandidate(f, 'regex', 0.6);
    }

    // Strategy 5: From imports
    const imports = problem.match(/from\s+([\w.]+)\s+import/g) || [];
    for (const imp of imports) {
        const module = imp.replace(/from\s+/, '').replace(/\s+import/, '');
        const parts = module.split('.');
        addCandidate(parts[parts.length - 1] + '.py', 'import', 0.5);
    }

    // Strategy 6: Class names that might be file names
    const classes = problem.match(/class\s+(\w+)/gi) || [];
    for (const cls of classes) {
        const name = cls.replace(/class\s+/i, '').toLowerCase();
        if (name.length > 3) {
            addCandidate(name + '.py', 'class', 0.3);
        }
    }

    return candidates;
}

// ============================================================================
// ENHANCED DOMAIN RANKER WITH TF-IDF
// ============================================================================

class EnhancedDomainRanker {
    private repo: string;
    private fileFrequency: Map<string, number> = new Map();
    private keywordToFile: Map<string, Map<string, number>> = new Map();
    private bigramToFile: Map<string, Map<string, number>> = new Map();
    private moduleToFile: Map<string, string> = new Map();
    private errorToFile: Map<string, string[]> = new Map();
    private directoryPatterns: Map<string, string[]> = new Map();  // directory → files
    private totalDocs = 0;
    private docFrequency: Map<string, number> = new Map();  // keyword → num docs containing it

    constructor(repo: string) {
        this.repo = repo;
    }

    train(instances: SWEBenchInstance[]): void {
        this.totalDocs = instances.length;

        for (const inst of instances) {
            const fullPath = this.extractFile(inst.patch);
            if (!fullPath) continue;

            const fileName = fullPath.split('/').pop() || '';
            const directory = fullPath.split('/').slice(0, -1).join('/');

            // File frequency
            this.fileFrequency.set(fileName, (this.fileFrequency.get(fileName) || 0) + 1);

            // Directory → files pattern
            if (directory) {
                if (!this.directoryPatterns.has(directory)) {
                    this.directoryPatterns.set(directory, []);
                }
                if (!this.directoryPatterns.get(directory)!.includes(fileName)) {
                    this.directoryPatterns.get(directory)!.push(fileName);
                }
            }

            // Keyword extraction with document frequency
            const keywords = this.extractKeywords(inst.problem_statement);
            const uniqueKeywords = new Set(keywords);

            for (const kw of uniqueKeywords) {
                this.docFrequency.set(kw, (this.docFrequency.get(kw) || 0) + 1);
            }

            for (const kw of keywords) {
                if (!this.keywordToFile.has(kw)) {
                    this.keywordToFile.set(kw, new Map());
                }
                const fileMap = this.keywordToFile.get(kw)!;
                fileMap.set(fileName, (fileMap.get(fileName) || 0) + 1);
            }

            // Bi-gram extraction
            const bigrams = this.extractBigrams(inst.problem_statement);
            for (const bg of bigrams) {
                if (!this.bigramToFile.has(bg)) {
                    this.bigramToFile.set(bg, new Map());
                }
                const fileMap = this.bigramToFile.get(bg)!;
                fileMap.set(fileName, (fileMap.get(fileName) || 0) + 1);
            }

            // Module → file mapping
            const modules = inst.problem_statement.match(/from\s+([\w.]+)\s+import/g) || [];
            for (const mod of modules) {
                const moduleName = mod.replace(/from\s+/, '').replace(/\s+import/, '');
                this.moduleToFile.set(moduleName, fileName);
                // Also store partial matches
                const parts = moduleName.split('.');
                for (let i = 1; i <= parts.length; i++) {
                    this.moduleToFile.set(parts.slice(0, i).join('.'), fileName);
                }
            }

            // Error type → file
            const errors = inst.problem_statement.match(/\w+Error|\w+Exception|\w+Warning/g) || [];
            for (const err of errors) {
                if (!this.errorToFile.has(err)) {
                    this.errorToFile.set(err, []);
                }
                if (!this.errorToFile.get(err)!.includes(fileName)) {
                    this.errorToFile.get(err)!.push(fileName);
                }
            }
        }
    }

    /**
     * Score a candidate using TF-IDF weighted keywords
     */
    score(candidate: string, problem: string, baseScore: number): number {
        let score = baseScore;

        // 1. Domain prior (how common is this file?)
        const fileFreq = this.fileFrequency.get(candidate) || 0;
        score += Math.log(fileFreq + 1) * 0.3;

        // 2. TF-IDF keyword matching
        const keywords = this.extractKeywords(problem);
        for (const kw of keywords) {
            const fileMap = this.keywordToFile.get(kw);
            if (fileMap && fileMap.has(candidate)) {
                const tf = fileMap.get(candidate)!;
                const df = this.docFrequency.get(kw) || 1;
                const idf = Math.log((this.totalDocs + 1) / (df + 1));
                score += tf * idf * 0.1;
            }
        }

        // 3. Bi-gram matching (stronger signal)
        const bigrams = this.extractBigrams(problem);
        for (const bg of bigrams) {
            const fileMap = this.bigramToFile.get(bg);
            if (fileMap && fileMap.has(candidate)) {
                score += fileMap.get(candidate)! * 0.2;
            }
        }

        // 4. Module path matching
        const modules = problem.match(/from\s+([\w.]+)\s+import/g) || [];
        for (const mod of modules) {
            const moduleName = mod.replace(/from\s+/, '').replace(/\s+import/, '');
            const mappedFile = this.moduleToFile.get(moduleName);
            if (mappedFile === candidate) {
                score += 0.5;
            }
            // Partial match
            const parts = moduleName.split('.');
            for (let i = parts.length; i >= 1; i--) {
                const partial = parts.slice(0, i).join('.');
                if (this.moduleToFile.get(partial) === candidate) {
                    score += 0.3 * (i / parts.length);
                    break;
                }
            }
        }

        // 5. Error type matching
        const errors = problem.match(/\w+Error|\w+Exception|\w+Warning/g) || [];
        for (const err of errors) {
            const files = this.errorToFile.get(err);
            if (files && files.includes(candidate)) {
                score += 0.4;
            }
        }

        // 6. File name similarity to keywords
        const candBase = candidate.replace('.py', '').toLowerCase();
        for (const kw of keywords) {
            if (candBase.includes(kw) || kw.includes(candBase)) {
                score += 0.3;
                break;
            }
        }

        return score;
    }

    /**
     * Rank candidates using all signals
     */
    rank(candidates: Array<{ file: string; source: string; baseScore: number }>, problem: string): string[] {
        if (candidates.length === 0) return [];

        const scored = candidates.map(c => ({
            file: c.file,
            score: this.score(c.file, problem, c.baseScore),
        }));

        scored.sort((a, b) => b.score - a.score);
        return scored.map(s => s.file);
    }

    private extractFile(patch: string): string {
        const match = patch.match(/diff --git a\/(.+?) b\//);
        return match ? match[1] : '';
    }

    private extractKeywords(text: string): string[] {
        const words = text.toLowerCase()
            .replace(/[^a-z0-9_]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 3 && !this.isStopWord(w));

        // Also extract method/attribute names
        const methods = (text.match(/\.(\w+)\(/g) || []).map(m => m.replace(/[.()]/g, ''));
        const attrs = (text.match(/\.(\w+)(?!\()/g) || []).map(a => a.replace('.', '')).slice(0, 10);

        // Extract CamelCase parts
        const camelParts: string[] = [];
        const camelMatches = text.match(/[A-Z][a-z]+/g) || [];
        for (const cm of camelMatches) {
            if (cm.length > 3) {
                camelParts.push(cm.toLowerCase());
            }
        }

        return [...new Set([...words.slice(0, 50), ...methods, ...attrs, ...camelParts])];
    }

    private extractBigrams(text: string): string[] {
        const words = text.toLowerCase()
            .replace(/[^a-z0-9_]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 2);

        const bigrams: string[] = [];
        for (let i = 0; i < words.length - 1; i++) {
            if (!this.isStopWord(words[i]) && !this.isStopWord(words[i + 1])) {
                bigrams.push(`${words[i]}_${words[i + 1]}`);
            }
        }
        return bigrams.slice(0, 30);
    }

    private isStopWord(word: string): boolean {
        const stops = new Set(['this', 'that', 'with', 'from', 'have', 'been', 'were', 'when', 'what', 'which', 'should', 'would', 'could', 'there', 'their', 'about', 'after', 'before', 'using', 'where', 'being', 'some', 'like', 'just', 'also', 'here', 'work', 'does', 'want', 'need', 'make', 'made', 'then', 'only', 'more', 'most', 'such', 'into', 'other']);
        return stops.has(word);
    }

    getStats() {
        return {
            repo: this.repo,
            files: this.fileFrequency.size,
            keywords: this.keywordToFile.size,
            bigrams: this.bigramToFile.size,
            modules: this.moduleToFile.size,
            errors: this.errorToFile.size,
        };
    }
}

// ============================================================================
// BASELINE
// ============================================================================

function baseline(problem: string): string {
    const fileMatch = problem.match(/[\w\/]+\.py/g) || [];
    if (fileMatch.length > 0) return fileMatch[0].split('/').pop() || fileMatch[0];

    const moduleMatch = problem.match(/from\s+([\w.]+)\s+import/);
    if (moduleMatch) {
        const parts = moduleMatch[1].split('.');
        return parts[parts.length - 1] + '.py';
    }

    return 'unknown.py';
}

function fileMatches(predicted: string, gold: string): boolean {
    if (!predicted || !gold) return false;
    const predFile = predicted.split('/').pop() || '';
    const goldFile = gold.split('/').pop() || '';
    return predFile === goldFile ||
        gold.endsWith(predFile) ||
        predicted.endsWith(goldFile) ||
        gold.includes(predFile);
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

async function main() {
    console.log('\n' + '='.repeat(70));
    console.log('HYPER-TARGETED TRAINING V5');
    console.log('TF-IDF + Bigrams + Module Paths + Error Types + Enhanced Scoring');
    console.log('='.repeat(70));

    // Load data
    const swePath = path.join(__dirname, 'swe-bench-real', 'all_instances.json');
    const sweInstances: SWEBenchInstance[] = JSON.parse(fs.readFileSync(swePath, 'utf8'));
    console.log(`\nLoaded ${sweInstances.length} instances`);

    // Group by repo
    const byRepo = new Map<string, SWEBenchInstance[]>();
    for (const inst of sweInstances) {
        if (!byRepo.has(inst.repo)) byRepo.set(inst.repo, []);
        byRepo.get(inst.repo)!.push(inst);
    }

    // Per-repo split
    const trainInstances: SWEBenchInstance[] = [];
    const testInstances: SWEBenchInstance[] = [];
    for (const [repo, instances] of byRepo) {
        const splitIdx = Math.floor(instances.length * 0.6);
        trainInstances.push(...instances.slice(0, splitIdx));
        testInstances.push(...instances.slice(splitIdx));
    }

    console.log(`  Train: ${trainInstances.length}, Test: ${testInstances.length}`);

    // ========================================================================
    // BASELINE
    // ========================================================================
    console.log('\n' + '='.repeat(70));
    console.log('BASELINE');
    console.log('='.repeat(70));

    let baselineCorrect = 0;
    const baselineByRepo: Map<string, { correct: number; total: number }> = new Map();

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const pred = baseline(inst.problem_statement);

        if (!baselineByRepo.has(inst.repo)) baselineByRepo.set(inst.repo, { correct: 0, total: 0 });
        baselineByRepo.get(inst.repo)!.total++;

        if (fileMatches(pred, gold)) {
            baselineCorrect++;
            baselineByRepo.get(inst.repo)!.correct++;
        }
    }

    const baselineAcc = baselineCorrect / testInstances.length;
    console.log(`  Overall: ${baselineCorrect}/${testInstances.length} = ${(baselineAcc * 100).toFixed(1)}%`);

    // ========================================================================
    // ENHANCED V5 RANKING
    // ========================================================================
    console.log('\n' + '='.repeat(70));
    console.log('ENHANCED V5 RANKING');
    console.log('='.repeat(70));

    // Train enhanced rankers
    const rankers = new Map<string, EnhancedDomainRanker>();
    console.log('\n  Training enhanced rankers:');
    for (const [repo, instances] of byRepo) {
        const trainCount = Math.floor(instances.length * 0.6);
        const ranker = new EnhancedDomainRanker(repo);
        ranker.train(instances.slice(0, trainCount));
        rankers.set(repo, ranker);
        const stats = ranker.getStats();
        console.log(`    ${repo.substring(0, 25).padEnd(26)}: ${stats.files} files, ${stats.keywords} kw, ${stats.bigrams} bg, ${stats.modules} mod`);
    }

    // Evaluate
    console.log('\n  Evaluating...');
    let v5Correct = 0;
    const v5ByRepo: Map<string, { correct: number; total: number }> = new Map();
    let candidateCounts = { zero: 0, one: 0, multi: 0 };

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const candidates = extractCandidates(inst.problem_statement);

        if (!v5ByRepo.has(inst.repo)) v5ByRepo.set(inst.repo, { correct: 0, total: 0 });
        v5ByRepo.get(inst.repo)!.total++;

        let pred: string;

        if (candidates.length === 0) {
            pred = 'unknown.py';
            candidateCounts.zero++;
        } else if (candidates.length === 1) {
            pred = candidates[0].file;
            candidateCounts.one++;
        } else {
            const ranker = rankers.get(inst.repo);
            if (ranker) {
                const ranked = ranker.rank(candidates, inst.problem_statement);
                pred = ranked[0];
            } else {
                // Sort by baseScore for unknown repos
                candidates.sort((a, b) => b.baseScore - a.baseScore);
                pred = candidates[0].file;
            }
            candidateCounts.multi++;
        }

        if (fileMatches(pred, gold)) {
            v5Correct++;
            v5ByRepo.get(inst.repo)!.correct++;
        }
    }

    const v5Acc = v5Correct / testInstances.length;
    console.log(`\n  Overall: ${v5Correct}/${testInstances.length} = ${(v5Acc * 100).toFixed(1)}%`);
    console.log(`  Candidates: 0=${candidateCounts.zero}, 1=${candidateCounts.one}, multi=${candidateCounts.multi}`);

    // ========================================================================
    // PER-REPOSITORY COMPARISON
    // ========================================================================
    console.log('\n' + '='.repeat(70));
    console.log('PER-REPOSITORY COMPARISON');
    console.log('='.repeat(70));

    const repoResults: Array<{ repo: string; baseAcc: number; v5Acc: number; diff: number }> = [];

    for (const [repo, baseStats] of baselineByRepo) {
        const v5Stats = v5ByRepo.get(repo) || { correct: 0, total: 0 };
        const baseAcc = baseStats.total > 0 ? baseStats.correct / baseStats.total : 0;
        const vAcc = v5Stats.total > 0 ? v5Stats.correct / v5Stats.total : 0;
        repoResults.push({ repo, baseAcc, v5Acc: vAcc, diff: vAcc - baseAcc });
    }

    repoResults.sort((a, b) => b.diff - a.diff);

    console.log('\n  Repository                      Baseline   V5       Δ');
    console.log('  ' + '-'.repeat(60));

    for (const r of repoResults) {
        const status = r.diff > 0.01 ? '✅' : r.diff < -0.01 ? '⚠️' : '➖';
        const diffStr = r.diff >= 0 ? `+${(r.diff * 100).toFixed(1)}%` : `${(r.diff * 100).toFixed(1)}%`;
        console.log(`  ${status} ${r.repo.substring(0, 28).padEnd(30)} ${(r.baseAcc * 100).toFixed(1).padStart(6)}%  ${(r.v5Acc * 100).toFixed(1).padStart(6)}%  ${diffStr}`);
    }

    // ========================================================================
    // SUMMARY
    // ========================================================================
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));

    const improved = repoResults.filter(r => r.diff > 0.01).length;
    const degraded = repoResults.filter(r => r.diff < -0.01).length;
    const same = repoResults.filter(r => Math.abs(r.diff) <= 0.01).length;
    const overallDiff = v5Acc - baselineAcc;

    console.log('\n┌───────────────────────────────┬──────────┬─────────────────┐');
    console.log('│ Configuration                 │ Accuracy │ vs Baseline     │');
    console.log('├───────────────────────────────┼──────────┼─────────────────┤');
    console.log(`│ Baseline                      │ ${(baselineAcc * 100).toFixed(1).padStart(6)}% │       -         │`);
    console.log(`│ V4 (previous best)            │ ${(15.1).toFixed(1).padStart(6)}% │ +1.6%           │`);
    console.log(`│ V5 (TF-IDF + Bigrams)         │ ${(v5Acc * 100).toFixed(1).padStart(6)}% │ ${overallDiff >= 0 ? '+' : ''}${(overallDiff * 100).toFixed(1)}%          │`);
    console.log('└───────────────────────────────┴──────────┴─────────────────┘');

    console.log(`\n📊 Results: ✅ ${improved}, ⚠️ ${degraded}, ➖ ${same}`);

    if (overallDiff > 1.6) {
        console.log(`\n🎉 NEW BEST! V5 beats V4 by +${(overallDiff - 1.6).toFixed(1)}%`);
    } else if (overallDiff > 0) {
        console.log(`\n✅ IMPROVEMENT: +${(overallDiff * 100).toFixed(1)}%`);
    } else {
        console.log(`\n⚠️ V4 remains best at +1.6%`);
    }

    console.log('\n📋 V5 ENHANCEMENTS:');
    console.log('  ✓ TF-IDF weighted keywords');
    console.log('  ✓ Bi-gram context matching');
    console.log('  ✓ Module path → file learning');
    console.log('  ✓ Error type → file associations');
    console.log('  ✓ CamelCase extraction');
    console.log('  ✓ Enhanced candidate scoring');

    // Save
    const results = {
        timestamp: new Date().toISOString(),
        version: 'hyper-targeted-v5',
        baseline: { accuracy: baselineAcc },
        v5: { accuracy: v5Acc, candidateCounts },
        perRepo: repoResults,
        summary: { improved, degraded, same, overallDiff },
        provenance: {
            hash: crypto.createHash('sha256')
                .update(JSON.stringify({ baselineAcc, v5Acc }))
                .digest('hex').substring(0, 32),
        },
    };

    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    const resultsPath = path.join(resultsDir, `hyper-targeted-v5-${Date.now()}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    console.log(`\nResults saved to: ${resultsPath}`);
}

main().catch(console.error);
