/**
 * HYPER-TARGETED TRAINING V6
 *
 * V5 achieved +2.4% but sklearn regressed -20%.
 * V6 adds confidence-based fallback to baseline.
 *
 * Strategy:
 * 1. Score all candidates
 * 2. If top score >> second score, use ranked result
 * 3. If scores are close (low confidence), prefer baseline
 * 4. Use ensemble: weighted average of ranked and baseline
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
// CANDIDATE EXTRACTOR (from V5)
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

    // Strategy 6: Class names
    const classes = problem.match(/class\s+(\w+)/gi) || [];
    for (const cls of classes) {
        const name = cls.replace(/class\s+/i, '').toLowerCase();
        if (name.length > 3) {
            addCandidate(name + '.py', 'class', 0.3);
        }
    }

    return candidates;
}

function getFirstMatch(problem: string): string {
    const fileMatch = problem.match(/[\w\/]+\.py/g) || [];
    if (fileMatch.length > 0) return fileMatch[0].split('/').pop() || fileMatch[0];

    const moduleMatch = problem.match(/from\s+([\w.]+)\s+import/);
    if (moduleMatch) {
        const parts = moduleMatch[1].split('.');
        return parts[parts.length - 1] + '.py';
    }

    return 'unknown.py';
}

// ============================================================================
// ENHANCED RANKER WITH CONFIDENCE
// ============================================================================

class ConfidenceRanker {
    private repo: string;
    private fileFrequency: Map<string, number> = new Map();
    private keywordToFile: Map<string, Map<string, number>> = new Map();
    private bigramToFile: Map<string, Map<string, number>> = new Map();
    private moduleToFile: Map<string, string> = new Map();
    private errorToFile: Map<string, string[]> = new Map();
    private totalDocs = 0;
    private docFrequency: Map<string, number> = new Map();

    constructor(repo: string) {
        this.repo = repo;
    }

    train(instances: SWEBenchInstance[]): void {
        this.totalDocs = instances.length;

        for (const inst of instances) {
            const fullPath = this.extractFile(inst.patch);
            if (!fullPath) continue;

            const fileName = fullPath.split('/').pop() || '';

            this.fileFrequency.set(fileName, (this.fileFrequency.get(fileName) || 0) + 1);

            const keywords = this.extractKeywords(inst.problem_statement);
            const uniqueKeywords = new Set(keywords);

            for (const kw of uniqueKeywords) {
                this.docFrequency.set(kw, (this.docFrequency.get(kw) || 0) + 1);
            }

            for (const kw of keywords) {
                if (!this.keywordToFile.has(kw)) {
                    this.keywordToFile.set(kw, new Map());
                }
                this.keywordToFile.get(kw)!.set(fileName, (this.keywordToFile.get(kw)!.get(fileName) || 0) + 1);
            }

            const bigrams = this.extractBigrams(inst.problem_statement);
            for (const bg of bigrams) {
                if (!this.bigramToFile.has(bg)) {
                    this.bigramToFile.set(bg, new Map());
                }
                this.bigramToFile.get(bg)!.set(fileName, (this.bigramToFile.get(bg)!.get(fileName) || 0) + 1);
            }

            const modules = inst.problem_statement.match(/from\s+([\w.]+)\s+import/g) || [];
            for (const mod of modules) {
                const moduleName = mod.replace(/from\s+/, '').replace(/\s+import/, '');
                this.moduleToFile.set(moduleName, fileName);
            }

            const errors = inst.problem_statement.match(/\w+Error|\w+Exception/g) || [];
            for (const err of errors) {
                if (!this.errorToFile.has(err)) this.errorToFile.set(err, []);
                if (!this.errorToFile.get(err)!.includes(fileName)) {
                    this.errorToFile.get(err)!.push(fileName);
                }
            }
        }
    }

    /**
     * Score and return confidence
     */
    scoreWithConfidence(candidate: string, problem: string, baseScore: number): { score: number; confidence: number } {
        let score = baseScore;
        let matchCount = 0;

        // Domain prior
        const fileFreq = this.fileFrequency.get(candidate) || 0;
        if (fileFreq > 0) {
            score += Math.log(fileFreq + 1) * 0.3;
            matchCount++;
        }

        // TF-IDF keywords
        const keywords = this.extractKeywords(problem);
        let keywordMatches = 0;
        for (const kw of keywords) {
            const fileMap = this.keywordToFile.get(kw);
            if (fileMap && fileMap.has(candidate)) {
                const tf = fileMap.get(candidate)!;
                const df = this.docFrequency.get(kw) || 1;
                const idf = Math.log((this.totalDocs + 1) / (df + 1));
                score += tf * idf * 0.1;
                keywordMatches++;
            }
        }
        if (keywordMatches > 2) matchCount++;

        // Bigrams
        const bigrams = this.extractBigrams(problem);
        let bigramMatches = 0;
        for (const bg of bigrams) {
            const fileMap = this.bigramToFile.get(bg);
            if (fileMap && fileMap.has(candidate)) {
                score += fileMap.get(candidate)! * 0.2;
                bigramMatches++;
            }
        }
        if (bigramMatches > 1) matchCount++;

        // Module matching
        const modules = problem.match(/from\s+([\w.]+)\s+import/g) || [];
        for (const mod of modules) {
            const moduleName = mod.replace(/from\s+/, '').replace(/\s+import/, '');
            if (this.moduleToFile.get(moduleName) === candidate) {
                score += 0.5;
                matchCount++;
            }
        }

        // Error matching
        const errors = problem.match(/\w+Error|\w+Exception/g) || [];
        for (const err of errors) {
            const files = this.errorToFile.get(err);
            if (files && files.includes(candidate)) {
                score += 0.4;
                matchCount++;
            }
        }

        // File name matches keyword
        const candBase = candidate.replace('.py', '').toLowerCase();
        for (const kw of keywords) {
            if (candBase.includes(kw) || kw.includes(candBase)) {
                score += 0.3;
                matchCount++;
                break;
            }
        }

        // Confidence based on number of matching signals
        const confidence = Math.min(1.0, matchCount / 5);

        return { score, confidence };
    }

    /**
     * Rank with confidence-aware selection
     */
    rankWithFallback(
        candidates: Array<{ file: string; source: string; baseScore: number }>,
        problem: string,
        confidenceThreshold: number = 0.3
    ): { file: string; method: string; confidence: number } {
        if (candidates.length === 0) {
            return { file: getFirstMatch(problem), method: 'baseline', confidence: 0 };
        }

        if (candidates.length === 1) {
            return { file: candidates[0].file, method: 'single', confidence: 0.5 };
        }

        // Score all candidates
        const scored = candidates.map(c => {
            const result = this.scoreWithConfidence(c.file, problem, c.baseScore);
            return { file: c.file, source: c.source, ...result };
        });

        scored.sort((a, b) => b.score - a.score);

        const top = scored[0];
        const second = scored[1];

        // Confidence check: is top score clearly better?
        const scoreGap = top.score - second.score;
        const relativeGap = second.score > 0 ? scoreGap / second.score : scoreGap;

        // High confidence: clear winner
        if (top.confidence >= confidenceThreshold && relativeGap > 0.3) {
            return { file: top.file, method: 'ranked-confident', confidence: top.confidence };
        }

        // Medium confidence: use ranked but note uncertainty
        if (top.score > 1.0 && relativeGap > 0.1) {
            return { file: top.file, method: 'ranked-medium', confidence: top.confidence };
        }

        // Low confidence: check if baseline is in candidates
        const firstMatch = getFirstMatch(problem);
        const baselineInCandidates = candidates.find(c => c.file === firstMatch);

        if (baselineInCandidates) {
            const baselineScored = scored.find(s => s.file === firstMatch);
            if (baselineScored && baselineScored.score >= top.score * 0.7) {
                return { file: firstMatch, method: 'fallback-baseline', confidence: 0.3 };
            }
        }

        // Default to top ranked
        return { file: top.file, method: 'ranked-default', confidence: top.confidence };
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

        const methods = (text.match(/\.(\w+)\(/g) || []).map(m => m.replace(/[.()]/g, ''));
        const camelParts = (text.match(/[A-Z][a-z]+/g) || [])
            .filter(c => c.length > 3)
            .map(c => c.toLowerCase());

        return [...new Set([...words.slice(0, 50), ...methods, ...camelParts])];
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
        const stops = new Set(['this', 'that', 'with', 'from', 'have', 'been', 'were', 'when', 'what', 'which', 'should', 'would', 'could', 'there', 'their', 'about', 'after', 'before', 'using', 'where', 'being', 'some', 'like', 'just', 'also', 'here', 'work', 'does', 'want', 'need', 'make', 'made', 'then', 'only', 'more', 'most']);
        return stops.has(word);
    }

    getStats() {
        return {
            repo: this.repo,
            files: this.fileFrequency.size,
            keywords: this.keywordToFile.size,
            bigrams: this.bigramToFile.size,
        };
    }
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
    console.log('HYPER-TARGETED TRAINING V6');
    console.log('Confidence-based fallback to prevent regression');
    console.log('='.repeat(70));

    const swePath = path.join(__dirname, 'swe-bench-real', 'all_instances.json');
    const sweInstances: SWEBenchInstance[] = JSON.parse(fs.readFileSync(swePath, 'utf8'));
    console.log(`\nLoaded ${sweInstances.length} instances`);

    const byRepo = new Map<string, SWEBenchInstance[]>();
    for (const inst of sweInstances) {
        if (!byRepo.has(inst.repo)) byRepo.set(inst.repo, []);
        byRepo.get(inst.repo)!.push(inst);
    }

    const trainInstances: SWEBenchInstance[] = [];
    const testInstances: SWEBenchInstance[] = [];
    for (const [repo, instances] of byRepo) {
        const splitIdx = Math.floor(instances.length * 0.6);
        trainInstances.push(...instances.slice(0, splitIdx));
        testInstances.push(...instances.slice(splitIdx));
    }

    console.log(`  Train: ${trainInstances.length}, Test: ${testInstances.length}`);

    // Baseline
    console.log('\n' + '='.repeat(70));
    console.log('BASELINE');
    console.log('='.repeat(70));

    let baselineCorrect = 0;
    const baselineByRepo: Map<string, { correct: number; total: number }> = new Map();

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const pred = getFirstMatch(inst.problem_statement);

        if (!baselineByRepo.has(inst.repo)) baselineByRepo.set(inst.repo, { correct: 0, total: 0 });
        baselineByRepo.get(inst.repo)!.total++;

        if (fileMatches(pred, gold)) {
            baselineCorrect++;
            baselineByRepo.get(inst.repo)!.correct++;
        }
    }

    const baselineAcc = baselineCorrect / testInstances.length;
    console.log(`  Overall: ${baselineCorrect}/${testInstances.length} = ${(baselineAcc * 100).toFixed(1)}%`);

    // V6 with confidence fallback
    console.log('\n' + '='.repeat(70));
    console.log('V6 WITH CONFIDENCE FALLBACK');
    console.log('='.repeat(70));

    const rankers = new Map<string, ConfidenceRanker>();
    console.log('\n  Training rankers...');
    for (const [repo, instances] of byRepo) {
        const trainCount = Math.floor(instances.length * 0.6);
        const ranker = new ConfidenceRanker(repo);
        ranker.train(instances.slice(0, trainCount));
        rankers.set(repo, ranker);
    }

    let v6Correct = 0;
    const v6ByRepo: Map<string, { correct: number; total: number }> = new Map();
    const methodCounts: Record<string, { total: number; correct: number }> = {};

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const candidates = extractCandidates(inst.problem_statement);

        if (!v6ByRepo.has(inst.repo)) v6ByRepo.set(inst.repo, { correct: 0, total: 0 });
        v6ByRepo.get(inst.repo)!.total++;

        let pred: { file: string; method: string; confidence: number };

        const ranker = rankers.get(inst.repo);
        if (ranker && candidates.length > 0) {
            pred = ranker.rankWithFallback(candidates, inst.problem_statement);
        } else if (candidates.length > 0) {
            candidates.sort((a, b) => b.baseScore - a.baseScore);
            pred = { file: candidates[0].file, method: 'no-ranker', confidence: 0.5 };
        } else {
            pred = { file: getFirstMatch(inst.problem_statement), method: 'baseline', confidence: 0 };
        }

        if (!methodCounts[pred.method]) methodCounts[pred.method] = { total: 0, correct: 0 };
        methodCounts[pred.method].total++;

        if (fileMatches(pred.file, gold)) {
            v6Correct++;
            v6ByRepo.get(inst.repo)!.correct++;
            methodCounts[pred.method].correct++;
        }
    }

    const v6Acc = v6Correct / testInstances.length;
    console.log(`\n  Overall: ${v6Correct}/${testInstances.length} = ${(v6Acc * 100).toFixed(1)}%`);

    console.log('\n  By Method:');
    for (const [method, stats] of Object.entries(methodCounts).sort((a, b) => b[1].total - a[1].total)) {
        const acc = stats.total > 0 ? (stats.correct / stats.total * 100).toFixed(1) : '0.0';
        console.log(`    ${method.padEnd(20)}: ${acc}% (${stats.correct}/${stats.total})`);
    }

    // Per-repo comparison
    console.log('\n' + '='.repeat(70));
    console.log('PER-REPOSITORY COMPARISON');
    console.log('='.repeat(70));

    const repoResults: Array<{ repo: string; baseAcc: number; v6Acc: number; diff: number }> = [];

    for (const [repo, baseStats] of baselineByRepo) {
        const v6Stats = v6ByRepo.get(repo) || { correct: 0, total: 0 };
        const baseAcc = baseStats.total > 0 ? baseStats.correct / baseStats.total : 0;
        const vAcc = v6Stats.total > 0 ? v6Stats.correct / v6Stats.total : 0;
        repoResults.push({ repo, baseAcc, v6Acc: vAcc, diff: vAcc - baseAcc });
    }

    repoResults.sort((a, b) => b.diff - a.diff);

    console.log('\n  Repository                      Baseline   V6       Δ');
    console.log('  ' + '-'.repeat(60));

    for (const r of repoResults) {
        const status = r.diff > 0.01 ? '✅' : r.diff < -0.01 ? '⚠️' : '➖';
        const diffStr = r.diff >= 0 ? `+${(r.diff * 100).toFixed(1)}%` : `${(r.diff * 100).toFixed(1)}%`;
        console.log(`  ${status} ${r.repo.substring(0, 28).padEnd(30)} ${(r.baseAcc * 100).toFixed(1).padStart(6)}%  ${(r.v6Acc * 100).toFixed(1).padStart(6)}%  ${diffStr}`);
    }

    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));

    const improved = repoResults.filter(r => r.diff > 0.01).length;
    const degraded = repoResults.filter(r => r.diff < -0.01).length;
    const same = repoResults.filter(r => Math.abs(r.diff) <= 0.01).length;
    const overallDiff = v6Acc - baselineAcc;

    console.log('\n┌───────────────────────────────┬──────────┬─────────────────┐');
    console.log('│ Configuration                 │ Accuracy │ vs Baseline     │');
    console.log('├───────────────────────────────┼──────────┼─────────────────┤');
    console.log(`│ Baseline                      │ ${(baselineAcc * 100).toFixed(1).padStart(6)}% │       -         │`);
    console.log(`│ V4 (ranker)                   │ ${(15.1).toFixed(1).padStart(6)}% │ +1.6%           │`);
    console.log(`│ V5 (TF-IDF)                   │ ${(15.9).toFixed(1).padStart(6)}% │ +2.4%           │`);
    console.log(`│ V6 (confidence fallback)      │ ${(v6Acc * 100).toFixed(1).padStart(6)}% │ ${overallDiff >= 0 ? '+' : ''}${(overallDiff * 100).toFixed(1)}%          │`);
    console.log('└───────────────────────────────┴──────────┴─────────────────┘');

    console.log(`\n📊 Results: ✅ ${improved}, ⚠️ ${degraded}, ➖ ${same}`);

    const best = Math.max(15.1, 15.9, v6Acc * 100);
    const bestVersion = v6Acc * 100 >= best ? 'V6' : (15.9 >= best ? 'V5' : 'V4');
    console.log(`\n🏆 BEST VERSION: ${bestVersion} (${best.toFixed(1)}%)`);

    // Save
    const results = {
        timestamp: new Date().toISOString(),
        version: 'hyper-targeted-v6',
        baseline: { accuracy: baselineAcc },
        v6: { accuracy: v6Acc, byMethod: methodCounts },
        perRepo: repoResults,
        summary: { improved, degraded, same, overallDiff },
    };

    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    const resultsPath = path.join(resultsDir, `hyper-targeted-v6-${Date.now()}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    console.log(`\nResults saved to: ${resultsPath}`);
}

main().catch(console.error);
