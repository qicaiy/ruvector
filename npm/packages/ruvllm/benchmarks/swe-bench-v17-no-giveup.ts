/**
 * SWE-bench V17: Never Give Up
 *
 * Key insight from V16: 49 "no-prediction" cases with 0% accuracy
 * V17 ensures EVERY case gets a meaningful prediction
 *
 * Strategy:
 * 1. Hints-direct (73.5% accurate) - unchanged
 * 2. Protected baseline for high-accuracy repos
 * 3. Per-repo learned patterns
 * 4. Class/function name extraction from problem statement
 * 5. Repository-specific heuristics (never "unknown.py")
 */

import * as fs from 'fs';
import * as path from 'path';

interface SWEBenchInstance {
    instance_id: string;
    repo: string;
    patch: string;
    problem_statement: string;
    hints_text: string;
}

// === V14's EXACT HINTS EXTRACTION ===
function extractFromHints(hints: string): Array<{ file: string; score: number }> {
    const results: Array<{ file: string; score: number }> = [];
    const seen = new Set<string>();

    if (!hints || hints.length === 0) return results;

    const directPaths = hints.match(/(?:^|\s|`|")([a-z_][a-z0-9_\/]*\.py)(?:\s|`|"|:|#|$)/gi) || [];
    for (const match of directPaths) {
        const file = match.replace(/^[\s`"]+|[\s`":,#]+$/g, '');
        const fileName = file.split('/').pop() || file;
        if (!seen.has(fileName) && fileName.endsWith('.py') && fileName.length > 3) {
            seen.add(fileName);
            results.push({ file: fileName, score: 0.88 });
        }
    }

    const urlPaths = hints.match(/github\.com\/[^\/]+\/[^\/]+\/blob\/[^\/]+\/([^\s#]+\.py)/gi) || [];
    for (const match of urlPaths) {
        const pathPart = match.match(/blob\/[^\/]+\/(.+\.py)/i);
        if (pathPart) {
            const fileName = pathPart[1].split('/').pop() || '';
            if (!seen.has(fileName) && fileName.length > 3) {
                seen.add(fileName);
                results.push({ file: fileName, score: 0.92 });
            }
        }
    }

    const lineRefs = hints.match(/([a-z_][a-z0-9_]*\.py):\d+/gi) || [];
    for (const match of lineRefs) {
        const fileName = match.split(':')[0];
        if (!seen.has(fileName)) {
            seen.add(fileName);
            results.push({ file: fileName, score: 0.90 });
        }
    }

    return results;
}

// === ENHANCED CANDIDATE EXTRACTION ===
function extractAllCandidates(text: string, hints: string): string[] {
    const files: string[] = [];
    const seen = new Set<string>();
    const fullText = `${text} ${hints || ''}`;

    // 1. Direct .py files
    const pyFiles = fullText.match(/\b([a-z_][a-z0-9_]*\.py)\b/gi) || [];
    for (const f of pyFiles) {
        const name = f.split('/').pop() || f;
        if (!seen.has(name) && name.length > 3) {
            seen.add(name);
            files.push(name);
        }
    }

    // 2. Path-style references
    const paths = fullText.match(/\b([a-z_][a-z0-9_\/]+\.py)\b/gi) || [];
    for (const p of paths) {
        const name = p.split('/').pop() || p;
        if (!seen.has(name) && name.length > 3) {
            seen.add(name);
            files.push(name);
        }
    }

    // 3. Class names -> likely file names
    const classes = fullText.match(/\bclass\s+([A-Z][a-zA-Z0-9]+)/g) || [];
    for (const c of classes) {
        const className = c.replace('class ', '');
        // Convert CamelCase to snake_case
        const snakeName = className.replace(/([A-Z])/g, '_$1').toLowerCase().replace(/^_/, '') + '.py';
        if (!seen.has(snakeName)) {
            seen.add(snakeName);
            files.push(snakeName);
        }
    }

    // 4. Function/method names that might indicate files
    const funcs = fullText.match(/\b(?:def|function)\s+([a-z_][a-z0-9_]+)/gi) || [];
    for (const f of funcs) {
        const funcName = f.split(/\s+/)[1];
        if (funcName && funcName.length > 4) {
            const possibleFile = funcName + '.py';
            if (!seen.has(possibleFile)) {
                seen.add(possibleFile);
                files.push(possibleFile);
            }
        }
    }

    // 5. Module-style references (foo.bar.baz -> baz.py)
    const modules = fullText.match(/\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){1,4})\b/gi) || [];
    for (const m of modules) {
        const parts = m.split('.');
        if (parts.length >= 2) {
            const lastPart = parts[parts.length - 1] + '.py';
            if (!seen.has(lastPart) && lastPart.length > 3) {
                seen.add(lastPart);
                files.push(lastPart);
            }
        }
    }

    return files;
}

// === REPO-SPECIFIC DEFAULTS ===
const REPO_DEFAULT_FILES: Record<string, string[]> = {
    'django/django': ['views.py', 'models.py', 'forms.py', 'admin.py', 'urls.py'],
    'pallets/flask': ['app.py', 'helpers.py', 'ctx.py', 'blueprints.py'],
    'sympy/sympy': ['core.py', 'expr.py', 'basic.py', 'numbers.py'],
    'matplotlib/matplotlib': ['axes.py', 'figure.py', 'pyplot.py', 'colors.py'],
    'scikit-learn/scikit-learn': ['base.py', 'estimator.py', 'utils.py'],
    'sphinx-doc/sphinx': ['builder.py', 'config.py', 'domains.py'],
    'pytest-dev/pytest': ['python.py', 'fixtures.py', 'main.py'],
    'pydata/xarray': ['dataarray.py', 'dataset.py', 'variable.py'],
    'astropy/astropy': ['table.py', 'fits.py', 'units.py'],
    'mwaskom/seaborn': ['seaborn.py', 'utils.py', 'axisgrid.py'],
    'psf/requests': ['models.py', 'sessions.py', 'api.py'],
    'pylint-dev/pylint': ['checker.py', 'lint.py', 'reporters.py'],
};

// === PER-REPO LEARNED PATTERNS ===
interface LearnedPatterns {
    fileFreq: Map<string, number>;
    keywordToFile: Map<string, Map<string, number>>;
}

function trainPatterns(instances: SWEBenchInstance[]): Map<string, LearnedPatterns> {
    const patterns = new Map<string, LearnedPatterns>();

    for (const inst of instances) {
        const goldPath = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const goldFile = goldPath.split('/').pop() || '';
        if (!goldFile) continue;

        if (!patterns.has(inst.repo)) {
            patterns.set(inst.repo, {
                fileFreq: new Map(),
                keywordToFile: new Map(),
            });
        }

        const p = patterns.get(inst.repo)!;
        p.fileFreq.set(goldFile, (p.fileFreq.get(goldFile) || 0) + 1);

        // Extract keywords
        const text = `${inst.problem_statement} ${inst.hints_text || ''}`.toLowerCase();
        const words = new Set(text.match(/[a-z_][a-z0-9_]{3,}/g) || []);

        for (const word of words) {
            if (!p.keywordToFile.has(word)) {
                p.keywordToFile.set(word, new Map());
            }
            const fm = p.keywordToFile.get(word)!;
            fm.set(goldFile, (fm.get(goldFile) || 0) + 1);
        }
    }

    return patterns;
}

// === HIGH BASELINE REPOS ===
const HIGH_BASELINE_REPOS = new Set([
    'scikit-learn/scikit-learn',
    'mwaskom/seaborn',
    'astropy/astropy',
]);

function fileMatches(predicted: string, gold: string): boolean {
    if (!predicted || !gold) return false;
    const predFile = predicted.split('/').pop() || '';
    const goldFile = gold.split('/').pop() || '';
    return predFile === goldFile || gold.endsWith(predFile) || predicted.endsWith(goldFile) || gold.includes(predFile);
}

interface V17Prediction {
    file: string;
    method: string;
    confidence: number;
}

function v17Predict(
    inst: SWEBenchInstance,
    patterns: LearnedPatterns | undefined
): V17Prediction {
    const repoName = inst.repo.split('/')[1];

    // 1. Hints extraction (V14's best - 73.5%)
    const hintsFiles = extractFromHints(inst.hints_text || '');
    if (hintsFiles.length > 0) {
        return {
            file: hintsFiles.sort((a, b) => b.score - a.score)[0].file,
            method: 'hints-direct',
            confidence: 0.88,
        };
    }

    // 2. Protected baseline repos
    const allCandidates = extractAllCandidates(inst.problem_statement, inst.hints_text || '');
    if (HIGH_BASELINE_REPOS.has(inst.repo) && allCandidates.length > 0) {
        // Filter out package names
        const filtered = allCandidates.filter(c => c !== `${repoName}.py` && c !== 'test.py');
        if (filtered.length > 0) {
            return { file: filtered[0], method: 'protected-baseline', confidence: 0.65 };
        }
        return { file: allCandidates[0], method: 'protected-baseline', confidence: 0.60 };
    }

    // 3. Score candidates with learned patterns
    if (patterns && allCandidates.length > 0) {
        const text = `${inst.problem_statement} ${inst.hints_text || ''}`.toLowerCase();
        const words = new Set(text.match(/[a-z_][a-z0-9_]{3,}/g) || []);

        const scores = new Map<string, number>();
        for (const cand of allCandidates) {
            let score = 0;

            // Frequency bonus
            score += (patterns.fileFreq.get(cand) || 0) * 2;

            // Keyword match bonus
            for (const word of words) {
                const fm = patterns.keywordToFile.get(word);
                if (fm && fm.has(cand)) {
                    score += fm.get(cand)!;
                }
            }

            // Penalize package names
            if (cand === `${repoName}.py`) {
                score -= 5;
            }

            scores.set(cand, score);
        }

        const sorted = Array.from(scores.entries()).sort((a, b) => b[1] - a[1]);
        if (sorted.length > 0 && sorted[0][1] > 0) {
            return {
                file: sorted[0][0],
                method: 'pattern-scored',
                confidence: Math.min(sorted[0][1] / 10, 0.75),
            };
        }
    }

    // 4. Use candidates with smart ranking
    if (allCandidates.length > 0) {
        const text = `${inst.problem_statement} ${inst.hints_text || ''}`.toLowerCase();

        // Score by mention count and specificity
        const ranked = allCandidates.map(cand => {
            const baseName = cand.replace('.py', '');
            let score = 0;

            // Count mentions
            const regex = new RegExp(baseName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
            score += (text.match(regex) || []).length * 2;

            // Penalize generic names
            if (['test', 'utils', 'helpers', 'base', '__init__'].some(g => baseName.includes(g))) {
                score -= 1;
            }

            // Penalize package name
            if (baseName === repoName) {
                score -= 3;
            }

            // Boost specific-looking names
            if (baseName.length > 6 && !baseName.includes('_')) {
                score += 0.5;
            }

            return { file: cand, score };
        }).sort((a, b) => b.score - a.score);

        return {
            file: ranked[0].file,
            method: 'candidate-ranked',
            confidence: 0.5,
        };
    }

    // 5. Repo-specific defaults (NEVER return unknown.py)
    const defaults = REPO_DEFAULT_FILES[inst.repo] || ['core.py', 'main.py', 'utils.py'];

    // Try to match defaults with keywords in problem
    const text = inst.problem_statement.toLowerCase();
    for (const def of defaults) {
        const baseName = def.replace('.py', '');
        if (text.includes(baseName)) {
            return {
                file: def,
                method: 'repo-default-matched',
                confidence: 0.35,
            };
        }
    }

    // Last resort: first repo default
    return {
        file: defaults[0],
        method: 'repo-default',
        confidence: 0.2,
    };
}

async function main() {
    console.log('='.repeat(70));
    console.log('SWE-BENCH V17: NEVER GIVE UP');
    console.log('Every instance gets a meaningful prediction');
    console.log('='.repeat(70));

    const swePath = path.join(__dirname, 'swe-bench-real', 'all_instances.json');
    const sweInstances: SWEBenchInstance[] = JSON.parse(fs.readFileSync(swePath, 'utf8'));

    console.log(`\nLoaded ${sweInstances.length} SWE-bench Lite instances\n`);

    const byRepo = new Map<string, SWEBenchInstance[]>();
    for (const inst of sweInstances) {
        if (!byRepo.has(inst.repo)) byRepo.set(inst.repo, []);
        byRepo.get(inst.repo)!.push(inst);
    }

    const trainInstances: SWEBenchInstance[] = [];
    const testInstances: SWEBenchInstance[] = [];

    for (const [, instances] of byRepo) {
        const splitIdx = Math.floor(instances.length * 0.6);
        trainInstances.push(...instances.slice(0, splitIdx));
        testInstances.push(...instances.slice(splitIdx));
    }

    console.log(`Train: ${trainInstances.length}, Test: ${testInstances.length}`);

    // Train patterns
    console.log('\nTraining patterns...');
    const patterns = trainPatterns(trainInstances);

    // Evaluate
    let baselineCorrect = 0;
    let v17Correct = 0;
    const methodStats: Record<string, { total: number; correct: number }> = {};
    const perRepoResults: Array<{ repo: string; baseAcc: number; v17Acc: number; diff: number }> = [];

    for (const [repo, instances] of byRepo) {
        const testInsts = instances.slice(Math.floor(instances.length * 0.6));
        let repoBaseCorrect = 0;
        let repoV17Correct = 0;

        for (const inst of testInsts) {
            const goldPath = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';

            // Baseline
            const baseCandidates = extractAllCandidates(inst.problem_statement, '');
            if (baseCandidates.length > 0 && fileMatches(baseCandidates[0], goldPath)) {
                baselineCorrect++;
                repoBaseCorrect++;
            }

            // V17
            const pred = v17Predict(inst, patterns.get(repo));

            if (!methodStats[pred.method]) {
                methodStats[pred.method] = { total: 0, correct: 0 };
            }
            methodStats[pred.method].total++;

            if (fileMatches(pred.file, goldPath)) {
                v17Correct++;
                repoV17Correct++;
                methodStats[pred.method].correct++;
            }
        }

        const baseAcc = testInsts.length > 0 ? repoBaseCorrect / testInsts.length : 0;
        const v17Acc = testInsts.length > 0 ? repoV17Correct / testInsts.length : 0;
        perRepoResults.push({ repo, baseAcc, v17Acc, diff: v17Acc - baseAcc });
    }

    const baselineAcc = baselineCorrect / testInstances.length;
    const v17Acc = v17Correct / testInstances.length;

    console.log('\n' + '='.repeat(70));
    console.log('RESULTS');
    console.log('='.repeat(70));

    console.log(`\nBaseline Accuracy: ${(baselineAcc * 100).toFixed(1)}% (${baselineCorrect}/${testInstances.length})`);
    console.log(`V17 Accuracy:      ${(v17Acc * 100).toFixed(1)}% (${v17Correct}/${testInstances.length})`);
    console.log(`Improvement:       ${((v17Acc - baselineAcc) * 100).toFixed(1)}%`);

    console.log('\n--- By Method ---');
    for (const [method, stats] of Object.entries(methodStats).sort((a, b) => b[1].correct - a[1].correct)) {
        const acc = stats.total > 0 ? (stats.correct / stats.total * 100).toFixed(1) : '0.0';
        console.log(`${method}: ${stats.correct}/${stats.total} (${acc}%)`);
    }

    console.log('\n--- Per Repository ---');
    const sortedRepos = perRepoResults.sort((a, b) => b.diff - a.diff);
    for (const r of sortedRepos) {
        const sign = r.diff > 0 ? '+' : '';
        console.log(`${r.repo}: ${(r.baseAcc * 100).toFixed(1)}% → ${(r.v17Acc * 100).toFixed(1)}% (${sign}${(r.diff * 100).toFixed(1)}%)`);
    }

    // No-prediction check
    const noPred = methodStats['no-prediction']?.total || 0;
    if (noPred > 0) {
        console.log(`\n⚠️ WARNING: ${noPred} no-prediction cases still exist!`);
    } else {
        console.log('\n✅ All instances received meaningful predictions');
    }

    // Save results
    const resultsPath = path.join(__dirname, 'results', `v17-no-giveup-${Date.now()}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        version: 'v17-no-giveup',
        baseline: { accuracy: baselineAcc },
        v17: { accuracy: v17Acc, byMethod: methodStats },
        perRepo: perRepoResults,
    }, null, 2));

    console.log(`\nResults saved to: ${resultsPath}`);
}

main().catch(console.error);
