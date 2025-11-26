#!/usr/bin/env python3
"""
Tisserand Parameter Analysis for Kuiper Belt Objects
Relative to Hypothetical Planet at 500 AU

Tisserand Parameter: T = (a_p/a) + 2*sqrt((a/a_p)*(1-e²))*cos(i)
Where:
  - a_p = 500 AU (planet semi-major axis)
  - a = object semi-major axis
  - e = eccentricity
  - i = inclination (degrees converted to radians)

Objects with similar T values are dynamically linked and may have
experienced gravitational interactions with the planet or common origin.
"""

import math
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict

@dataclass
class KBO:
    """Kuiper Belt Object"""
    name: str
    a: float  # semi-major axis (AU)
    e: float  # eccentricity
    i: float  # inclination (degrees)
    q: float  # perihelion distance (AU)
    ad: float  # aphelion distance (AU)
    period: float  # orbital period (days)
    h: float  # absolute magnitude
    class_name: str  # TNO classification

    def __post_init__(self):
        self.tisserand = None
        self.group = None


class TisserandAnalyzer:
    """Analyzes Tisserand parameters and dynamical linking"""

    def __init__(self, planet_au: float = 500.0):
        """Initialize analyzer with planet at specified semi-major axis"""
        self.planet_au = planet_au
        self.objects: List[KBO] = []
        self.groups: Dict[int, List[KBO]] = defaultdict(list)

    def load_kbo_data(self) -> None:
        """Load KBO data from the Rust module output"""
        self.objects = [
            # DWARF PLANETS AND MAJOR TNOs
            KBO("134340 Pluto", 39.59, 0.2518, 17.15, 29.619, 49.56, 91000.0, -0.54, "TNO"),
            KBO("136199 Eris", 68.0, 0.4370, 43.87, 38.284, 97.71, 205000.0, -1.25, "TNO"),
            KBO("136108 Haumea", 43.01, 0.1958, 28.21, 34.586, 51.42, 103000.0, 0.14, "TNO"),
            KBO("136472 Makemake", 45.51, 0.1604, 29.03, 38.210, 52.81, 112000.0, -0.25, "TNO"),
            KBO("225088 Gonggong", 66.89, 0.5032, 30.87, 33.235, 100.55, 200000.0, 1.84, "TNO"),
            KBO("90377 Sedna", 549.5, 0.8613, 11.93, 76.223, 1022.86, 4710000.0, 1.49, "TNO"),
            KBO("50000 Quaoar", 43.15, 0.0358, 7.99, 41.601, 44.69, 104000.0, 2.41, "TNO"),
            KBO("90482 Orcus", 39.34, 0.2217, 20.56, 30.614, 48.06, 90100.0, 2.14, "TNO"),

            # PLUTINOS (3:2 NEPTUNE RESONANCE ~39.4 AU)
            KBO("15810 Arawn", 39.21, 0.1141, 3.81, 34.734, 43.68, 89700.0, 7.68, "TNO"),
            KBO("28978 Ixion", 39.35, 0.2442, 19.67, 29.740, 48.96, 90200.0, 3.47, "TNO"),
            KBO("38628 Huya", 39.21, 0.2729, 15.48, 28.513, 49.91, 89700.0, 4.79, "TNO"),
            KBO("47171 Lempo", 39.72, 0.2298, 8.40, 30.591, 48.85, 91400.0, 4.93, "TNO"),
            KBO("208996 Achlys", 39.63, 0.1748, 13.55, 32.699, 46.56, 91100.0, 3.72, "TNO"),
            KBO("84922 (2003 VS2)", 39.71, 0.0816, 14.76, 36.476, 42.95, 91400.0, 3.99, "TNO"),
            KBO("455502 (2003 UZ413)", 39.43, 0.2182, 12.04, 30.824, 48.03, 90400.0, 4.27, "TNO"),

            # CLASSICAL KUIPER BELT (Cubewanos)
            KBO("15760 Albion", 44.2, 0.0725, 2.19, 40.995, 47.40, 107000.0, 7.18, "TNO"),
            KBO("20000 Varuna", 43.18, 0.0525, 17.14, 40.909, 45.45, 104000.0, 3.79, "TNO"),
            KBO("19521 Chaos", 46.11, 0.1105, 12.02, 41.013, 51.20, 114000.0, 4.63, "TNO"),
            KBO("79360 Sila-Nunam", 44.04, 0.0141, 2.24, 43.415, 44.66, 107000.0, 5.26, "TNO"),
            KBO("66652 Borasisi", 43.79, 0.0849, 0.56, 40.075, 47.51, 106000.0, 5.86, "TNO"),
            KBO("58534 Logos", 45.23, 0.1227, 2.90, 39.681, 50.79, 111000.0, 6.87, "TNO"),

            # SCATTERED DISK OBJECTS
            KBO("15874 (1996 TL66)", 84.89, 0.5866, 23.96, 35.094, 134.69, 286000.0, 5.41, "TNO"),
            KBO("26181 (1996 GQ21)", 92.48, 0.5874, 13.36, 38.152, 146.81, 325000.0, 4.84, "TNO"),
            KBO("26375 (1999 DE9)", 55.5, 0.4201, 7.61, 32.184, 78.81, 151000.0, 4.89, "TNO"),
            KBO("82075 (2000 YW134)", 58.23, 0.2936, 19.77, 41.128, 75.32, 162000.0, 4.65, "TNO"),
            KBO("84522 (2002 TC302)", 55.84, 0.2995, 35.01, 39.113, 72.56, 152000.0, 3.92, "TNO"),

            # EXTREME/DETACHED OBJECTS
            KBO("148209 (2000 CR105)", 228.7, 0.8071, 22.71, 44.117, 413.29, 1260000.0, 6.14, "TNO"),
            KBO("82158 (2001 FP185)", 213.4, 0.8398, 30.80, 34.190, 392.66, 1140000.0, 6.16, "TNO"),
            KBO("87269 (2000 OO67)", 617.9, 0.9663, 20.05, 20.850, 1215.04, 5610000.0, 9.10, "TNO"),
            KBO("308933 (2006 SQ372)", 839.3, 0.9711, 19.46, 24.226, 1654.33, 8880000.0, 7.94, "TNO"),
            KBO("445473 (2010 VZ98)", 159.8, 0.7851, 4.51, 34.356, 285.32, 738000.0, 5.04, "TNO"),

            # ADDITIONAL OBJECTS
            KBO("65407 (2002 RP120)", 54.53, 0.9542, 119.37, 2.498, 106.57, 147000.0, 12.43, "TNO"),
            KBO("127546 (2002 XU93)", 66.9, 0.6862, 77.95, 20.991, 112.80, 200000.0, 8.06, "TNO"),
            KBO("336756 (2010 NV1)", 305.2, 0.9690, 140.82, 9.457, 600.93, 1950000.0, 10.55, "TNO"),
            KBO("418993 (2009 MS9)", 375.7, 0.9706, 67.96, 11.046, 740.43, 2660000.0, 9.74, "TNO"),
        ]

    def calculate_tisserand(self, obj: KBO) -> float:
        """
        Calculate Tisserand parameter relative to hypothetical planet.

        T = (a_p/a) + 2*sqrt((a/a_p)*(1-e²))*cos(i)
        """
        a_p = self.planet_au
        a = obj.a
        e = obj.e
        i_rad = math.radians(obj.i)  # Convert degrees to radians

        # Calculate components
        first_term = a_p / a

        # (a/a_p) * (1 - e²)
        factor = (a / a_p) * (1 - e**2)

        # sqrt((a/a_p)*(1-e²)) * cos(i)
        second_term = 2 * math.sqrt(factor) * math.cos(i_rad)

        tisserand = first_term + second_term
        return tisserand

    def analyze(self) -> None:
        """Perform complete Tisserand analysis"""
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║     TISSERAND PARAMETER ANALYSIS FOR KUIPER BELT OBJECTS      ║")
        print("║         Relative to Hypothetical Planet at 500 AU            ║")
        print("╚════════════════════════════════════════════════════════════════╝\n")

        # Calculate Tisserand parameter for each object
        for obj in self.objects:
            obj.tisserand = self.calculate_tisserand(obj)

        # Sort by Tisserand parameter
        self.objects.sort(key=lambda x: x.tisserand)

        # Display detailed results
        self._display_results()

        # Group objects by Tisserand similarity
        self._group_by_tisserand()

        # Display groups
        self._display_groups()

        # Analyze dynamical families
        self._analyze_families()

    def _display_results(self) -> None:
        """Display individual Tisserand calculations"""
        print("📊 TISSERAND PARAMETER CALCULATIONS\n")
        print("─" * 100)
        print(f"{'Object Name':<40} {'a (AU)':<12} {'e':<10} {'i (°)':<10} {'T_500AU':<15}")
        print("─" * 100)

        for obj in self.objects:
            print(f"{obj.name:<40} {obj.a:<12.2f} {obj.e:<10.4f} {obj.i:<10.2f} {obj.tisserand:<15.6f}")

        print("─" * 100)
        print(f"\nRange: T = {min(obj.tisserand for obj in self.objects):.6f} to {max(obj.tisserand for obj in self.objects):.6f}\n")

    def _group_by_tisserand(self, threshold: float = 0.5) -> None:
        """
        Group objects by similar Tisserand parameters.
        Objects with |ΔT| < threshold are considered dynamically linked.
        """
        used = set()
        group_id = 0

        for i, obj1 in enumerate(self.objects):
            if i in used:
                continue

            group = [obj1]
            used.add(i)
            obj1.group = group_id

            # Find all objects within threshold of this one
            for j in range(i + 1, len(self.objects)):
                if j in used:
                    continue
                obj2 = self.objects[j]

                if abs(obj1.tisserand - obj2.tisserand) < threshold:
                    group.append(obj2)
                    used.add(j)
                    obj2.group = group_id
                else:
                    break  # Objects are sorted, so we can stop

            self.groups[group_id] = group
            group_id += 1

    def _display_groups(self) -> None:
        """Display dynamical groups"""
        print("\n📍 DYNAMICAL GROUPS (ΔT < 0.5)\n")
        print("Objects with similar Tisserand parameters are dynamically linked\n")

        for group_id in sorted(self.groups.keys()):
            group = self.groups[group_id]

            if len(group) > 1:
                # Display group header
                t_values = [obj.tisserand for obj in group]
                avg_t = sum(t_values) / len(t_values)

                print(f"\n🔗 GROUP {group_id + 1}: {len(group)} objects (avg T = {avg_t:.6f})")
                print("─" * 100)

                for obj in sorted(group, key=lambda x: x.tisserand):
                    print(f"  {obj.name:<35} T = {obj.tisserand:>10.6f}  "
                          f"(a={obj.a:>8.2f}, e={obj.e:>7.4f}, i={obj.i:>7.2f}°)")
            else:
                # Isolated objects
                obj = group[0]
                print(f"\n🎯 ISOLATED: {obj.name:<35} T = {obj.tisserand:>10.6f}  "
                      f"(a={obj.a:>8.2f}, e={obj.e:>7.4f}, i={obj.i:>7.2f}°)")

    def _analyze_families(self) -> None:
        """Analyze dynamical families and populations"""
        print("\n\n📈 DYNAMICAL POPULATION ANALYSIS\n")
        print("=" * 100)

        # Count populations
        grouped = sum(1 for g in self.groups.values() if len(g) > 1)
        isolated = sum(1 for g in self.groups.values() if len(g) == 1)
        total_grouped = sum(len(g) for g in self.groups.values() if len(g) > 1)

        print(f"Total Objects Analyzed:     {len(self.objects)}")
        print(f"Dynamical Groups Found:     {grouped} (with 2+ members)")
        print(f"Isolated Objects:           {isolated}")
        print(f"Objects in Groups:          {total_grouped}")
        print("\n" + "=" * 100)

        # Identify significant populations
        print("\n\n🌟 SIGNIFICANT DYNAMICAL POPULATIONS\n")

        populations = []
        for group_id, group in sorted(self.groups.items()):
            if len(group) > 1:
                t_values = [obj.tisserand for obj in group]
                avg_t = sum(t_values) / len(t_values)
                t_spread = max(t_values) - min(t_values)

                populations.append({
                    'id': group_id,
                    'size': len(group),
                    'avg_t': avg_t,
                    'spread': t_spread,
                    'objects': group
                })

        # Sort by group size
        populations.sort(key=lambda x: x['size'], reverse=True)

        for pop in populations:
            print(f"\n✓ Population {pop['id'] + 1}: {pop['size']} objects")
            print(f"  Average Tisserand:  {pop['avg_t']:.6f}")
            print(f"  T Spread:           {pop['spread']:.6f} (range represents orbital coherence)")
            print(f"  Members:")
            for obj in pop['objects']:
                print(f"    • {obj.name:<35} (T={obj.tisserand:.6f})")

        # Analyze dynamical coherence
        print("\n\n🔬 DYNAMICAL COHERENCE METRICS\n")
        print("─" * 100)

        for pop in populations:
            group = pop['objects']

            # Calculate orbital statistics
            a_values = [obj.a for obj in group]
            e_values = [obj.e for obj in group]
            i_values = [obj.i for obj in group]

            a_mean = sum(a_values) / len(a_values)
            e_mean = sum(e_values) / len(e_values)
            i_mean = sum(i_values) / len(i_values)

            a_spread = max(a_values) - min(a_values)
            e_spread = max(e_values) - min(e_values)
            i_spread = max(i_values) - min(i_values)

            print(f"\nGroup {pop['id'] + 1}:")
            print(f"  Semi-major axis:  mean={a_mean:.2f} AU,  spread={a_spread:.2f} AU")
            print(f"  Eccentricity:     mean={e_mean:.4f},    spread={e_spread:.4f}")
            print(f"  Inclination:      mean={i_mean:.2f}°,   spread={i_spread:.2f}°")
            print(f"  Tisserand spread: {pop['spread']:.6f}")

            # Interpret coherence
            if pop['spread'] < 0.1:
                coherence = "Very High"
            elif pop['spread'] < 0.3:
                coherence = "High"
            elif pop['spread'] < 0.5:
                coherence = "Moderate"
            else:
                coherence = "Low"

            print(f"  Dynamical Coherence: {coherence}")

    def export_results(self, filename: str = "tisserand_results.json") -> None:
        """Export results to JSON"""
        results = {
            'planet_au': self.planet_au,
            'analysis_date': '2025-11-26',
            'objects': [
                {
                    'name': obj.name,
                    'a': obj.a,
                    'e': obj.e,
                    'i': obj.i,
                    'tisserand': obj.tisserand,
                    'group': obj.group,
                    'class': obj.class_name
                }
                for obj in self.objects
            ],
            'groups': {
                str(gid): [obj.name for obj in group]
                for gid, group in self.groups.items()
            }
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results exported to {filename}\n")


def main():
    """Main analysis routine"""
    analyzer = TisserandAnalyzer(planet_au=500.0)
    analyzer.load_kbo_data()
    analyzer.analyze()
    analyzer.export_results("/home/user/ruvector/examples/kuiper_belt/tisserand_results.json")

    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║                   ANALYSIS COMPLETE                           ║")
    print("║                                                                ║")
    print("║  Interpretation Guide:                                        ║")
    print("║  - Tisserand parameter is conserved in gravitational close   ║")
    print("║    encounters (invariant of motion)                          ║")
    print("║  - Objects with similar T may share common origin or have    ║")
    print("║    experienced encounters with the hypothetical planet      ║")
    print("║  - Lower T spread → higher dynamical coherence              ║")
    print("║  - Detected populations suggest collisional families or     ║")
    print("║    mean-motion resonance populations                        ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
