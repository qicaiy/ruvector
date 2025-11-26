/// Perihelion Clustering Analysis for Planet Nine Detection
/// Analyzes ETNOs (Extreme Trans-Neptunian Objects) for perihelion longitude clustering
/// This is the key signature of gravitational perturbation from an undiscovered massive planet

use std::f64::consts::PI;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TNO {
    pub name: String,
    pub a: f64,      // Semi-major axis (AU)
    pub e: f64,      // Eccentricity
    pub q: f64,      // Perihelion distance (AU)
    pub omega: f64,  // Longitude of ascending node (degrees)
    pub w: f64,      // Argument of perihelion (degrees)
}

impl TNO {
    /// Calculate longitude of perihelion (varpi = omega + w)
    pub fn longitude_of_perihelion(&self) -> f64 {
        (self.omega + self.w) % 360.0
    }
}

#[derive(Debug, Clone)]
pub struct PerihelionClusteringResult {
    pub extreme_tno_count: usize,
    pub etno_objects: Vec<String>,
    pub perihelia_longitudes: Vec<f64>,
    pub mean_longitude: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub is_clustered: bool,
    pub clustering_confidence: f64,
    pub statistical_significance: f64,
    pub estimated_perturber_longitude: f64,
    pub rayleigh_test: f64,
    pub uniformity_p_value: f64,
}

/// Analyze perihelion clustering in Kuiper Belt objects
pub fn analyze_perihelion_clustering(objects: &[TNO]) -> PerihelionClusteringResult {
    // Filter for extreme TNOs: a > 150 AU and q > 30 AU
    let extreme_tnos: Vec<_> = objects.iter()
        .filter(|o| o.a > 150.0 && o.q > 30.0)
        .collect();

    let mut result = PerihelionClusteringResult {
        extreme_tno_count: extreme_tnos.len(),
        etno_objects: extreme_tnos.iter().map(|o| o.name.clone()).collect(),
        perihelia_longitudes: Vec::new(),
        mean_longitude: 0.0,
        std_dev: 0.0,
        variance: 0.0,
        is_clustered: false,
        clustering_confidence: 0.0,
        statistical_significance: 0.0,
        estimated_perturber_longitude: 0.0,
        rayleigh_test: 0.0,
        uniformity_p_value: 0.0,
    };

    if extreme_tnos.is_empty() {
        return result;
    }

    // Calculate perihelion longitudes for each ETNO
    let perihelia: Vec<f64> = extreme_tnos.iter()
        .map(|o| o.longitude_of_perihelion())
        .collect();

    result.perihelia_longitudes = perihelia.clone();

    // Calculate circular mean (accounting for wraparound at 360°)
    let sin_sum: f64 = perihelia.iter()
        .map(|p| (p * PI / 180.0).sin())
        .sum();
    let cos_sum: f64 = perihelia.iter()
        .map(|p| (p * PI / 180.0).cos())
        .sum();

    let mean_rad = sin_sum.atan2(cos_sum);
    let mean_deg = mean_rad * 180.0 / PI;
    result.mean_longitude = if mean_deg < 0.0 { mean_deg + 360.0 } else { mean_deg };

    // Calculate resultant vector length (r) for Rayleigh test
    let r = ((sin_sum.powi(2) + cos_sum.powi(2)).sqrt()) / perihelia.len() as f64;
    result.rayleigh_test = r;

    // Rayleigh test for non-uniformity
    // Z = n * r^2 where n is sample size
    let z = perihelia.len() as f64 * r.powi(2);
    // Approximate p-value using exponential distribution
    let p_value = (-z).exp();
    result.uniformity_p_value = p_value;

    // Linear standard deviation for visualization (approximate)
    let deviations: Vec<f64> = perihelia.iter()
        .map(|p| {
            let diff = (p - result.mean_longitude).abs();
            if diff > 180.0 { 360.0 - diff } else { diff }
        })
        .collect();

    let variance = deviations.iter()
        .map(|d| d.powi(2))
        .sum::<f64>() / deviations.len() as f64;
    result.variance = variance;
    result.std_dev = variance.sqrt();

    // Clustering detection: std_dev < 90° indicates clustering
    // (random distribution would have std_dev ~ 104°)
    result.is_clustered = result.std_dev < 90.0;

    // Confidence score: 0 to 1
    // Based on both standard deviation and Rayleigh test
    result.clustering_confidence = ((1.0 - (result.std_dev / 180.0)).max(0.0)) * 0.5
                                  + r * 0.5;

    // Statistical significance: combine multiple metrics
    // p-value < 0.05 is generally considered significant
    result.statistical_significance = if p_value < 0.05 { 1.0 - p_value } else { 0.0 }
                                     + if result.is_clustered { 0.5 } else { 0.0 };

    // Estimated perturber is anti-aligned (opposite direction)
    result.estimated_perturber_longitude = (result.mean_longitude + 180.0) % 360.0;

    result
}

/// Calculate statistical clustering metrics
pub fn calculate_clustering_metrics(angles: &[f64]) -> ClusteringMetrics {
    if angles.is_empty() {
        return ClusteringMetrics::default();
    }

    // Convert to radians for circular statistics
    let angles_rad: Vec<f64> = angles.iter().map(|a| a * PI / 180.0).collect();

    // Circular mean
    let sin_sum: f64 = angles_rad.iter().map(|a| a.sin()).sum();
    let cos_sum: f64 = angles_rad.iter().map(|a| a.cos()).sum();
    let mean_rad = sin_sum.atan2(cos_sum);
    let mean_deg = mean_rad * 180.0 / PI;
    let mean = if mean_deg < 0.0 { mean_deg + 360.0 } else { mean_deg };

    // Resultant vector length (concentration parameter)
    let r = ((sin_sum.powi(2) + cos_sum.powi(2)).sqrt()) / angles.len() as f64;

    // Circular variance and standard deviation
    let circ_var = 1.0 - r;
    let circ_std_dev = (-2.0 * (1.0 - r).ln()).sqrt(); // in radians
    let circ_std_dev_deg = circ_std_dev * 180.0 / PI;

    // Rayleigh test statistic
    let z = angles.len() as f64 * r.powi(2);
    let p_value = (-z).exp();

    // Linear metrics
    let deviations: Vec<f64> = angles.iter()
        .map(|a| {
            let diff = (a - mean).abs();
            if diff > 180.0 { 360.0 - diff } else { diff }
        })
        .collect();

    let variance = deviations.iter()
        .map(|d| d.powi(2))
        .sum::<f64>() / deviations.len() as f64;
    let std_dev = variance.sqrt();

    ClusteringMetrics {
        n: angles.len(),
        mean: mean,
        circular_variance: circ_var,
        circular_std_dev: circ_std_dev_deg,
        linear_variance: variance,
        linear_std_dev: std_dev,
        concentration_r: r,
        rayleigh_z: z,
        p_value,
    }
}

#[derive(Debug, Default)]
pub struct ClusteringMetrics {
    pub n: usize,
    pub mean: f64,
    pub circular_variance: f64,
    pub circular_std_dev: f64,
    pub linear_variance: f64,
    pub linear_std_dev: f64,
    pub concentration_r: f64,
    pub rayleigh_z: f64,
    pub p_value: f64,
}

/// Perform clustered grouping analysis
pub fn group_by_longitude_clusters(angles: &[f64], cluster_size: f64) -> Vec<Vec<f64>> {
    if angles.is_empty() {
        return Vec::new();
    }

    let mut sorted_angles = angles.to_vec();
    sorted_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut clusters: Vec<Vec<f64>> = Vec::new();
    let mut current_cluster = vec![sorted_angles[0]];

    for &angle in &sorted_angles[1..] {
        let diff = angle - current_cluster.last().unwrap();
        if diff < cluster_size {
            current_cluster.push(angle);
        } else {
            clusters.push(current_cluster);
            current_cluster = vec![angle];
        }
    }
    clusters.push(current_cluster);

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longitude_of_perihelion() {
        let tno = TNO {
            name: "Test".to_string(),
            a: 200.0,
            e: 0.5,
            q: 50.0,
            omega: 100.0,
            w: 50.0,
        };
        let varpi = tno.longitude_of_perihelion();
        assert_eq!(varpi, 150.0);
    }

    #[test]
    fn test_perihelion_clustering_empty() {
        let objects: Vec<TNO> = vec![];
        let result = analyze_perihelion_clustering(&objects);
        assert_eq!(result.extreme_tno_count, 0);
        assert!(!result.is_clustered);
    }
}
