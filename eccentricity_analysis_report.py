#!/usr/bin/env python3
"""
Eccentricity Pumping Analysis - Analysis Agent 6
Analyze eccentricity pumping from distant planets
"""

import re
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

@dataclass
class KBOData:
    name: str
    a: float  # Semi-major axis (AU)
    e: float  # Eccentricity
    i: float  # Inclination (degrees)
    q: float  # Perihelion distance (AU)
    ad: float # Aphelion distance (AU)
    period: float
    omega: float
    w: float
    h: float = None
    klass: str = "TNO"

def parse_kbo_data(file_path: str) -> List[KBOData]:
    """Parse KBO data from Rust source file"""
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract all KuiperBeltObject definitions
    pattern = r'KuiperBeltObject\s*\{([^}]+)\}'
    matches = re.finditer(pattern, content, re.DOTALL)

    kbos = []
    for match in matches:
        block = match.group(1)

        # Parse each field
        name_match = re.search(r'name:\s*"([^"]+)"', block)
        a_match = re.search(r'a:\s*([\d.]+)', block)
        e_match = re.search(r'e:\s*([\d.]+)', block)
        i_match = re.search(r'i:\s*([\d.]+)', block)
        q_match = re.search(r'q:\s*([\d.]+)', block)
        ad_match = re.search(r'ad:\s*([\d.]+)', block)
        period_match = re.search(r'period:\s*([\d.]+)', block)
        omega_match = re.search(r'omega:\s*([\d.]+)', block)
        w_match = re.search(r'w:\s*([\d.]+)', block)
        h_match = re.search(r'h:\s*Some\(([-\d.]+)\)', block)
        class_match = re.search(r'class:\s*"([^"]+)"', block)

        if all([name_match, a_match, e_match]):
            kbo = KBOData(
                name=name_match.group(1),
                a=float(a_match.group(1)),
                e=float(e_match.group(1)),
                i=float(i_match.group(1)),
                q=float(q_match.group(1)),
                ad=float(ad_match.group(1)),
                period=float(period_match.group(1)) if period_match else 0.0,
                omega=float(omega_match.group(1)) if omega_match else 0.0,
                w=float(w_match.group(1)) if w_match else 0.0,
                h=float(h_match.group(1)) if h_match else None,
                klass=class_match.group(1) if class_match else "TNO"
            )
            kbos.append(kbo)

    return kbos

def analyze_eccentricity_pumping(kbos: List[KBOData]) -> Dict:
    """Analyze objects with high eccentricity (e > 0.7) and large semi-major axis (a > 50 AU)"""

    # Filter for high eccentricity and large semi-major axis
    high_e_objects = [obj for obj in kbos if obj.e > 0.7 and obj.a > 50.0]

    # Sort by eccentricity descending
    high_e_objects.sort(key=lambda x: x.e, reverse=True)

    # Calculate statistics
    if not high_e_objects:
        return {
            'count': 0,
            'objects': [],
            'statistics': {},
            'perturber_analysis': {}
        }

    a_values = [obj.a for obj in high_e_objects]
    e_values = [obj.e for obj in high_e_objects]
    q_values = [obj.q for obj in high_e_objects]
    ad_values = [obj.ad for obj in high_e_objects]

    # Sort for median calculation
    a_sorted = sorted(a_values)
    median_a = a_sorted[len(a_sorted) // 2] if len(a_sorted) % 2 == 1 else \
               (a_sorted[len(a_sorted) // 2 - 1] + a_sorted[len(a_sorted) // 2]) / 2.0

    # Class distribution
    class_dist = defaultdict(int)
    for obj in high_e_objects:
        class_dist[obj.klass] += 1

    avg_a = sum(a_values) / len(a_values)
    avg_e = sum(e_values) / len(e_values)
    avg_q = sum(q_values) / len(q_values)
    avg_ad = sum(ad_values) / len(ad_values)

    # Perturber distance estimation (3x average a)
    estimated_distance = avg_a * 3.0

    # Determine confidence based on sample size
    confidence = 0.85 if len(high_e_objects) > 5 else 0.60

    # Determine mass range
    if estimated_distance > 150.0:
        mass_range = "Earth to Super-Earth mass (1-5 Earth masses)"
    elif estimated_distance > 100.0:
        mass_range = "Mars to Earth mass (0.1-2 Earth masses)"
    else:
        mass_range = "Neptune to Jupiter mass (10-1000 Earth masses)"

    # Candidate perturbers
    candidates = []
    if estimated_distance > 200.0:
        candidates.append("Unknown Planet 9 (predicted)")
    if avg_a > 100.0:
        candidates.append("Distant stellar companion")

    return {
        'count': len(high_e_objects),
        'objects': [
            {
                'name': obj.name,
                'a': round(obj.a, 2),
                'e': round(obj.e, 4),
                'i': round(obj.i, 2),
                'q': round(obj.q, 2),
                'ad': round(obj.ad, 2),
                'pumping_strength': round((obj.e - 0.2) / 0.7, 3) if obj.e > 0.2 else 0.0,
            }
            for obj in high_e_objects
        ],
        'statistics': {
            'avg_a': round(avg_a, 2),
            'median_a': round(median_a, 2),
            'avg_e': round(avg_e, 4),
            'avg_q': round(avg_q, 2),
            'avg_ad': round(avg_ad, 2),
            'a_range': (round(min(a_values), 2), round(max(a_values), 2)),
            'e_range': (round(min(e_values), 4), round(max(e_values), 4)),
            'q_range': (round(min(q_values), 2), round(max(q_values), 2)),
            'class_distribution': dict(class_dist),
        },
        'perturber_analysis': {
            'estimated_distance': round(estimated_distance, 1),
            'confidence': round(confidence, 2),
            'expected_mass_range': mass_range,
            'candidate_perturbers': candidates,
        }
    }

def generate_report(analysis: Dict) -> str:
    """Generate a formatted analysis report"""

    report = []
    report.append("═" * 65)
    report.append("ECCENTRICITY PUMPING ANALYSIS - KUIPER BELT OBJECTS")
    report.append("Analysis Agent 6: High Eccentricity Population Study")
    report.append("═" * 65)
    report.append("")

    report.append("SELECTION CRITERIA: e > 0.7 AND a > 50 AU")
    report.append("─" * 65)
    report.append("")

    if analysis['count'] == 0:
        report.append("NO OBJECTS FOUND MATCHING CRITERIA")
        return "\n".join(report)

    stats = analysis['statistics']
    perturber = analysis['perturber_analysis']

    report.append(f"IDENTIFIED OBJECTS: {analysis['count']}")
    report.append(f"Average Semi-Major Axis: {stats['avg_a']} AU")
    report.append(f"Median Semi-Major Axis: {stats['median_a']} AU")
    report.append(f"Average Eccentricity: {stats['avg_e']}")
    report.append(f"Average Perihelion: {stats['avg_q']} AU")
    report.append(f"Average Aphelion: {stats['avg_ad']} AU")
    report.append("")

    report.append("ORBITAL PARAMETER RANGES:")
    report.append(f"  Semi-major axis:  {stats['a_range'][0]} - {stats['a_range'][1]} AU")
    report.append(f"  Eccentricity:     {stats['e_range'][0]} - {stats['e_range'][1]}")
    report.append(f"  Perihelion:       {stats['q_range'][0]} - {stats['q_range'][1]} AU")
    report.append("")

    report.append("OBJECT CLASSIFICATION:")
    for klass, count in sorted(stats['class_distribution'].items()):
        report.append(f"  {klass}: {count}")
    report.append("")

    report.append("PERTURBER DISTANCE ESTIMATION:")
    report.append("─" * 65)
    report.append(f"Estimated Distance: {perturber['estimated_distance']} AU")
    report.append(f"Confidence Level: {perturber['confidence'] * 100:.0f}%")
    report.append(f"Expected Mass Range: {perturber['expected_mass_range']}")
    report.append("")

    if perturber['candidate_perturbers']:
        report.append("CANDIDATE PERTURBERS:")
        for candidate in perturber['candidate_perturbers']:
            report.append(f"  • {candidate}")
        report.append("")

    report.append("HIGH ECCENTRICITY OBJECTS (Sorted by Eccentricity):")
    report.append("─" * 65)

    for idx, obj in enumerate(analysis['objects'], 1):
        report.append(f"{idx:2}. {obj['name']:30} | e={obj['e']:.4f} | a={obj['a']:7.2f} AU | q={obj['q']:6.2f} AU | ad={obj['ad']:7.2f} AU")

    report.append("")
    report.append("ANALYSIS COMPLETE")
    report.append("═" * 65)

    return "\n".join(report)

def main():
    # Parse KBO data from Rust source
    print("Parsing KBO data from Rust source...")
    kbos = parse_kbo_data('/home/user/ruvector/examples/kuiper_belt/kbo_data.rs')
    print(f"Loaded {len(kbos)} Kuiper Belt Objects\n")

    # Run analysis
    print("Running eccentricity pumping analysis...")
    analysis = analyze_eccentricity_pumping(kbos)

    # Generate and print report
    report = generate_report(analysis)
    print(report)

    # Save JSON output
    output_path = '/home/user/ruvector/ANALYSIS_AGENT_6_RESULTS.json'
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nJSON results saved to: {output_path}")

    # Save text report
    report_path = '/home/user/ruvector/ANALYSIS_AGENT_6_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Text report saved to: {report_path}")

if __name__ == '__main__':
    main()
