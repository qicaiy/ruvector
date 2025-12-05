/**
 * Analyze why flask and pylint are stuck at 0%
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

    return results;
}

function extractPyFiles(text: string): string[] {
    const matches = text.match(/\b([a-z_][a-z0-9_]*\.py)\b/gi) || [];
    return [...new Set(matches)];
}

async function main() {
    const swePath = path.join(__dirname, 'swe-bench-real', 'all_instances.json');
    const sweInstances: SWEBenchInstance[] = JSON.parse(fs.readFileSync(swePath, 'utf8'));

    const byRepo = new Map<string, SWEBenchInstance[]>();
    for (const inst of sweInstances) {
        if (!byRepo.has(inst.repo)) byRepo.set(inst.repo, []);
        byRepo.get(inst.repo)!.push(inst);
    }

    for (const repo of ['pallets/flask', 'pylint-dev/pylint']) {
        console.log('='.repeat(70));
        console.log(`REPO: ${repo}`);
        console.log('='.repeat(70));

        const instances = byRepo.get(repo) || [];
        const testInstances = instances.slice(Math.floor(instances.length * 0.6));

        console.log(`\nTest instances: ${testInstances.length}\n`);

        for (const inst of testInstances) {
            const goldPath = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
            const goldFile = goldPath.split('/').pop() || '';

            console.log(`--- ${inst.instance_id} ---`);
            console.log(`Gold: ${goldPath}`);

            // Check hints
            const hintsFiles = extractFromHints(inst.hints_text || '');
            console.log(`Hints extracted: ${hintsFiles.map(h => h.file).join(', ') || 'NONE'}`);

            // Check problem statement
            const problemFiles = extractPyFiles(inst.problem_statement);
            console.log(`Problem files: ${problemFiles.slice(0, 5).join(', ')}`);

            // Is gold in problem?
            const goldInProblem = problemFiles.some(f => f === goldFile || goldPath.includes(f));
            console.log(`Gold in problem: ${goldInProblem ? 'YES' : 'NO'}`);

            // Is gold in hints?
            const goldInHints = hintsFiles.some(h => h.file === goldFile || goldPath.includes(h.file));
            console.log(`Gold in hints: ${goldInHints ? 'YES' : 'NO'}`);

            // Show relevant parts of problem
            if (!goldInProblem && !goldInHints) {
                console.log(`\nProblem preview: "${inst.problem_statement.substring(0, 200)}..."`);
                if (inst.hints_text) {
                    console.log(`Hints preview: "${inst.hints_text.substring(0, 200)}..."`);
                }
            }

            console.log('');
        }
    }
}

main().catch(console.error);
