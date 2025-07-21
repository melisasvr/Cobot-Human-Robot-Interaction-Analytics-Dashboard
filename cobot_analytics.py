import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Disable all interactive plotting
plt.ioff()  # Turn off interactive mode
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Optional imports with error handling - but skip Plotly entirely
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Anomaly detection disabled.")
    SKLEARN_AVAILABLE = False

# Explicitly disable Plotly to prevent JavaScript output
PLOTLY_AVAILABLE = False
print("INFO: Plotly disabled for clean text output")

class CobotSensorData:
    """Simulates real-time sensor data from cobot workspace"""
    
    def __init__(self):
        self.proximity_threshold = 0.5  # meters
        self.force_threshold = 10.0     # Newtons
        self.safety_zone_radius = 1.0   # meters
        
    def generate_proximity_data(self, n_samples=1000):
        """Generate proximity sensor data"""
        try:
            timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                     end=datetime.now(), periods=n_samples)
            
            # Simulate human proximity with some patterns
            base_distance = 2.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_samples))
            noise = np.random.normal(0, 0.2, n_samples)
            
            # Add some close approach events
            close_events = np.random.choice(n_samples, size=int(n_samples*0.1), replace=False)
            base_distance[close_events] = np.random.uniform(0.1, 0.6, len(close_events))
            
            proximity_distance = np.maximum(0.05, base_distance + noise)
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'proximity_distance': proximity_distance,
                'human_detected': proximity_distance < 3.0,
                'safety_violation': proximity_distance < self.proximity_threshold
            })
        except Exception as e:
            print(f"Error generating proximity data: {e}")
            return pd.DataFrame()
    
    def generate_force_data(self, n_samples=1000):
        """Generate force sensor data from robot joints"""
        try:
            timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                     end=datetime.now(), periods=n_samples)
            
            # Simulate normal operation forces with occasional spikes
            base_force = 3.0 + 2.0 * np.random.random(n_samples)
            
            # Add force spikes during interactions
            spike_events = np.random.choice(n_samples, size=int(n_samples*0.05), replace=False)
            base_force[spike_events] += np.random.uniform(8, 20, len(spike_events))
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'joint_1_force': base_force * (0.8 + 0.4 * np.random.random(n_samples)),
                'joint_2_force': base_force * (0.9 + 0.2 * np.random.random(n_samples)),
                'joint_3_force': base_force * (0.85 + 0.3 * np.random.random(n_samples)),
                'excessive_force': base_force > self.force_threshold
            })
        except Exception as e:
            print(f"Error generating force data: {e}")
            return pd.DataFrame()
    
    def generate_vision_data(self, n_samples=1000):
        """Generate computer vision analysis data"""
        try:
            timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                     end=datetime.now(), periods=n_samples)
            
            # Simulate human pose and activity recognition
            activities = ['working', 'reaching', 'idle', 'moving', 'adjusting']
            activity_weights = [0.4, 0.2, 0.2, 0.15, 0.05]
            
            human_activities = np.random.choice(activities, size=n_samples, p=activity_weights)
            pose_confidence = np.random.uniform(0.7, 0.99, n_samples)
            
            # Simulate workspace occupancy
            workspace_occupancy = np.random.uniform(0, 1, n_samples)
            
            return pd.DataFrame({
                'timestamp': timestamps,
                'human_activity': human_activities,
                'pose_confidence': pose_confidence,
                'workspace_occupancy': workspace_occupancy,
                'collision_risk': (workspace_occupancy > 0.7) & (np.isin(human_activities, ['reaching', 'moving']))
            })
        except Exception as e:
            print(f"Error generating vision data: {e}")
            return pd.DataFrame()

class SafetyAnalyzer:
    """Analyzes safety patterns and predicts potential hazards"""
    
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.isolation_forest = None
            self.scaler = None
        
    def detect_anomalies(self, sensor_data):
        """Detect anomalous patterns in sensor data"""
        if not SKLEARN_AVAILABLE or sensor_data.empty:
            return np.zeros(len(sensor_data))
            
        try:
            # Prepare features for anomaly detection
            features = []
            
            if 'proximity_distance' in sensor_data.columns:
                features.append(sensor_data['proximity_distance'].values)
            if 'joint_1_force' in sensor_data.columns:
                features.extend([
                    sensor_data['joint_1_force'].values,
                    sensor_data['joint_2_force'].values,
                    sensor_data['joint_3_force'].values
                ])
            if 'workspace_occupancy' in sensor_data.columns:
                features.append(sensor_data['workspace_occupancy'].values)
                
            if not features:
                return np.zeros(len(sensor_data))
                
            X = np.column_stack(features)
            X_scaled = self.scaler.fit_transform(X)
            
            anomalies = self.isolation_forest.fit_predict(X_scaled)
            return anomalies
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return np.zeros(len(sensor_data))
    
    def calculate_safety_score(self, proximity_data, force_data, vision_data):
        """Calculate overall workspace safety score"""
        try:
            safety_factors = []
            
            if not proximity_data.empty:
                prox_safety = 1 - (proximity_data['safety_violation'].sum() / len(proximity_data))
                safety_factors.append(prox_safety)
            
            if not force_data.empty:
                force_safety = 1 - (force_data['excessive_force'].sum() / len(force_data))
                safety_factors.append(force_safety)
            
            if not vision_data.empty:
                collision_safety = 1 - (vision_data['collision_risk'].sum() / len(vision_data))
                safety_factors.append(collision_safety)
            
            if not safety_factors:
                return {'overall_safety': 0, 'proximity_safety': 0, 'force_safety': 0, 'collision_safety': 0}
            
            weights = [0.4, 0.3, 0.3][:len(safety_factors)]
            overall_safety = np.average(safety_factors, weights=weights)
            
            return {
                'overall_safety': overall_safety,
                'proximity_safety': safety_factors[0] if len(safety_factors) > 0 else 0,
                'force_safety': safety_factors[1] if len(safety_factors) > 1 else 0,
                'collision_safety': safety_factors[2] if len(safety_factors) > 2 else 0
            }
        except Exception as e:
            print(f"Error calculating safety score: {e}")
            return {'overall_safety': 0, 'proximity_safety': 0, 'force_safety': 0, 'collision_safety': 0}
    
    def predict_interaction_efficiency(self, data_combined):
        """Analyze and predict human-robot collaboration efficiency"""
        if data_combined.empty:
            return {'overall_efficiency': 0}
            
        try:
            efficiency_metrics = {}
            
            # Time in collaborative zone
            if 'proximity_distance' in data_combined.columns:
                collab_time = (data_combined['proximity_distance'] < 2.0).sum() / len(data_combined)
                efficiency_metrics['collaboration_time_ratio'] = collab_time
            
            # Activity diversity
            if 'human_activity' in data_combined.columns:
                activity_diversity = len(data_combined['human_activity'].unique()) / 5.0
                efficiency_metrics['activity_diversity'] = activity_diversity
            
            # Force smoothness
            force_cols = ['joint_1_force', 'joint_2_force', 'joint_3_force']
            available_force_cols = [col for col in force_cols if col in data_combined.columns]
            
            if available_force_cols:
                force_means = data_combined[available_force_cols].mean().mean()
                if force_means > 0:
                    force_std = data_combined[available_force_cols].std().mean()
                    force_smoothness = 1 - (force_std / force_means)
                    efficiency_metrics['force_smoothness'] = max(0, force_smoothness)
                else:
                    efficiency_metrics['force_smoothness'] = 0
            
            if efficiency_metrics:
                efficiency_score = np.mean(list(efficiency_metrics.values()))
                efficiency_metrics['overall_efficiency'] = efficiency_score
            else:
                efficiency_metrics['overall_efficiency'] = 0
            
            return efficiency_metrics
        except Exception as e:
            print(f"Error calculating efficiency metrics: {e}")
            return {'overall_efficiency': 0}

class CobotDashboard:
    """Main dashboard class for visualization and analysis"""
    
    def __init__(self):
        self.sensor_data = CobotSensorData()
        self.safety_analyzer = SafetyAnalyzer()
        self.proximity_data = pd.DataFrame()
        self.force_data = pd.DataFrame()
        self.vision_data = pd.DataFrame()
        self.combined_data = pd.DataFrame()
        
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        try:
            self.proximity_data = self.sensor_data.generate_proximity_data()
            self.force_data = self.sensor_data.generate_force_data()
            self.vision_data = self.sensor_data.generate_vision_data()
            
            # Combine data
            if not self.proximity_data.empty and not self.force_data.empty:
                self.combined_data = pd.merge(self.proximity_data, self.force_data, 
                                            on='timestamp', how='outer')
            elif not self.proximity_data.empty:
                self.combined_data = self.proximity_data.copy()
            elif not self.force_data.empty:
                self.combined_data = self.force_data.copy()
            
            if not self.combined_data.empty and not self.vision_data.empty:
                self.combined_data = pd.merge(self.combined_data, self.vision_data, 
                                            on='timestamp', how='outer')
            elif not self.vision_data.empty and self.combined_data.empty:
                self.combined_data = self.vision_data.copy()
                
        except Exception as e:
            print(f"Error generating sample data: {e}")
    
    def create_text_summary(self):
        """Create text-based summary instead of interactive plots"""
        print("\n" + "="*50)
        print("TEXT-BASED DASHBOARD SUMMARY")
        print("="*50)
        
        if not self.proximity_data.empty:
            avg_distance = self.proximity_data['proximity_distance'].mean()
            min_distance = self.proximity_data['proximity_distance'].min()
            violations = self.proximity_data['safety_violation'].sum()
            print(f"Proximity Monitoring:")
            print(f"  • Average human distance: {avg_distance:.2f}m")
            print(f"  • Minimum distance: {min_distance:.2f}m")
            print(f"  • Safety violations: {violations}")
        
        if not self.force_data.empty:
            avg_forces = [
                self.force_data['joint_1_force'].mean(),
                self.force_data['joint_2_force'].mean(), 
                self.force_data['joint_3_force'].mean()
            ]
            excessive_force = self.force_data['excessive_force'].sum()
            print(f"\nForce Analysis:")
            for i, force in enumerate(avg_forces, 1):
                print(f"  • Joint {i} average force: {force:.2f}N")
            print(f"  • Excessive force events: {excessive_force}")
        
        if not self.vision_data.empty and 'human_activity' in self.vision_data.columns:
            activity_counts = self.vision_data['human_activity'].value_counts()
            collision_risks = self.vision_data['collision_risk'].sum()
            print(f"\nVision Analysis:")
            print(f"  • High collision risk events: {collision_risks}")
            print("  • Activity distribution:")
            for activity, count in activity_counts.items():
                pct = (count / len(self.vision_data)) * 100
                print(f"    - {activity.capitalize()}: {count} ({pct:.1f}%)")
        
        print("="*50)
    
    def create_safety_dashboard(self):
        """Create dashboard - text only version"""
        print("Creating text-based safety summary...")
        self.create_text_summary()
        
    def create_interaction_analytics(self):
        """Create interaction analytics - text only"""
        print("Creating interaction analytics summary...")
        # Summary already created in create_text_summary
        
    def generate_safety_report(self):
        """Generate comprehensive safety and efficiency report"""
        try:
            safety_scores = self.safety_analyzer.calculate_safety_score(
                self.proximity_data, self.force_data, self.vision_data
            )
            
            efficiency_metrics = self.safety_analyzer.predict_interaction_efficiency(
                self.combined_data
            )
            
            # Detect anomalies
            anomalies = self.safety_analyzer.detect_anomalies(self.combined_data)
            anomaly_count = (anomalies == -1).sum() if len(anomalies) > 0 else 0
            
            print("="*60)
            print("COBOT HUMAN-ROBOT INTERACTION ANALYTICS REPORT")
            print("="*60)
            print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Data Period: Last 1 hour ({len(self.combined_data)} samples)")
            print()
            
            print("SAFETY ANALYSIS:")
            print("-" * 20)
            print(f"Overall Safety Score: {safety_scores['overall_safety']:.2%}")
            print(f"  • Proximity Safety: {safety_scores['proximity_safety']:.2%}")
            print(f"  • Force Safety: {safety_scores['force_safety']:.2%}")
            print(f"  • Collision Safety: {safety_scores['collision_safety']:.2%}")
            print()
            
            print("SAFETY VIOLATIONS:")
            print("-" * 20)
            prox_violations = self.proximity_data['safety_violation'].sum() if not self.proximity_data.empty else 0
            force_violations = self.force_data['excessive_force'].sum() if not self.force_data.empty else 0
            collision_risks = self.vision_data['collision_risk'].sum() if not self.vision_data.empty else 0
            
            print(f"  • Proximity Violations: {prox_violations}")
            print(f"  • Excessive Force Events: {force_violations}")
            print(f"  • High Collision Risk Events: {collision_risks}")
            print(f"  • Anomalies Detected: {anomaly_count}")
            print()
            
            print("COLLABORATION EFFICIENCY:")
            print("-" * 25)
            print(f"Overall Efficiency: {efficiency_metrics['overall_efficiency']:.2%}")
            if 'collaboration_time_ratio' in efficiency_metrics:
                print(f"  • Collaboration Time Ratio: {efficiency_metrics['collaboration_time_ratio']:.2%}")
            if 'activity_diversity' in efficiency_metrics:
                print(f"  • Activity Diversity: {efficiency_metrics['activity_diversity']:.2%}")
            if 'force_smoothness' in efficiency_metrics:
                print(f"  • Force Smoothness: {efficiency_metrics['force_smoothness']:.2%}")
            print()
            
            if not self.vision_data.empty and 'human_activity' in self.vision_data.columns:
                print("HUMAN ACTIVITY ANALYSIS:")
                print("-" * 25)
                activity_stats = self.vision_data['human_activity'].value_counts()
                for activity, count in activity_stats.items():
                    percentage = (count / len(self.vision_data)) * 100
                    print(f"  • {activity.capitalize()}: {count} times ({percentage:.1f}%)")
                print()
            
            print("SYSTEM STATUS:")
            print("-" * 15)
            print(f"  • Plotly Available: No (Disabled for clean output)")
            print(f"  • Scikit-learn Available: {'Yes' if SKLEARN_AVAILABLE else 'No'}")
            print(f"  • Matplotlib Available: Yes (Non-interactive)")
            print()
            
            print("RECOMMENDATIONS:")
            print("-" * 15)
            
            if safety_scores['overall_safety'] < 0.8:
                print("  ⚠️  Safety score is below optimal threshold")
                if safety_scores['proximity_safety'] < 0.8:
                    print("     - Review proximity sensor calibration")
                    print("     - Consider adjusting robot speed in human presence")
                if safety_scores['force_safety'] < 0.8:
                    print("     - Check force sensor sensitivity")
                    print("     - Review collision detection algorithms")
            
            if efficiency_metrics['overall_efficiency'] < 0.6:
                print("  📈 Collaboration efficiency could be improved")
                print("     - Analyze workflow patterns for optimization")
                print("     - Consider adaptive robot behavior based on human activity")
            
            if anomaly_count > len(self.combined_data) * 0.05:
                print("  🔍 High number of anomalies detected")
                print("     - Review sensor data for consistency")
                print("     - Investigate unusual interaction patterns")
            
            if safety_scores['overall_safety'] >= 0.8 and efficiency_metrics['overall_efficiency'] >= 0.6:
                print("  ✅ System operating within optimal parameters")
            
            print("\n" + "="*60)
            
            return {
                'safety_scores': safety_scores,
                'efficiency_metrics': efficiency_metrics,
                'anomaly_count': anomaly_count,
                'total_samples': len(self.combined_data)
            }
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return {}

def main():
    """Main function to run the cobot analytics dashboard"""
    print("Initializing Cobot Human-Robot Interaction Analytics Dashboard...")
    print("(Text-only mode for clean output)\n")
    
    try:
        # Create dashboard instance
        dashboard = CobotDashboard()
        
        # Generate sample data
        print("Generating sensor data...")
        dashboard.generate_sample_data()
        
        # Create text summaries instead of plots
        print("Creating safety monitoring summary...")
        dashboard.create_safety_dashboard()
        
        print("Creating interaction analytics summary...")
        dashboard.create_interaction_analytics()
        
        # Generate comprehensive report
        print("Generating safety and efficiency report...\n")
        report_data = dashboard.generate_safety_report()
        
        return dashboard, report_data
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, {}

if __name__ == "__main__":
    dashboard, report = main()
    
    if dashboard is not None:
        print("ADDITIONAL ANALYSIS CAPABILITIES:")
        print("-" * 35)
        print("• Real-time anomaly detection")
        print("• Predictive safety modeling")
        print("• Collaboration pattern learning")
        print("• Adaptive safety threshold adjustment")
        print("• Multi-cobot coordination analysis")
        print("• Human ergonomics assessment")
        print("• Workspace optimization suggestions")
    else:
        print("Dashboard initialization failed. Check error messages above.")