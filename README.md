# Cobot Human-Robot Interaction Analytics Dashboard
A comprehensive Python-based analytics system for monitoring and analyzing collaborative robot (cobot) safety, efficiency, and human-robot interaction patterns in real-time.

## Overview
This system simulates and analyzes sensor data from collaborative robots working alongside humans, providing safety monitoring, anomaly detection, and interaction efficiency analysis. The dashboard generates text-based reports and summaries for easy integration into industrial monitoring systems.

## Features
### 🛡️ Safety Monitoring
- **Proximity Detection**: Real-time human distance monitoring with configurable safety thresholds
- **Force Analysis**: Multi-joint force sensor monitoring to detect excessive forces
- **Vision Analysis**: Computer vision-based collision risk assessment
- **Safety Scoring**: Comprehensive safety metrics with weighted scoring system

### 🔍 Analytics & Intelligence
- **Anomaly Detection**: Machine learning-based detection of unusual interaction patterns
- **Predictive Analytics**: Efficiency prediction and optimization recommendations
- **Activity Recognition**: Human activity classification and workspace occupancy tracking
- **Pattern Analysis**: Historical trend analysis and workflow optimization insights

### 📊 Reporting
- **Real-time Dashboards**: Text-based summaries for clean integration
- **Comprehensive Reports**: Detailed safety and efficiency analysis
- **Customizable Metrics**: Configurable thresholds and scoring parameters
- **Recommendations Engine**: Automated suggestions for system optimization

## Installation

### Prerequisites
- Python 3.7+
- Required packages (install via pip):

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Optional Dependencies
- `scikit-learn`: Required for anomaly detection (automatic fallback if unavailable)
- `matplotlib`: Used for non-interactive plotting backend

## Quick Start

1. **Clone or download** the `cobot_analytics.py` file
2. **Run the script** directly:
```bash
python cobot_analytics.py
```

3. **View the output** in your terminal - the system will generate:
   - Real-time sensor data simulation
   - Safety monitoring summaries
   - Interaction analytics
   - Comprehensive safety and efficiency reports

## Usage

### Basic Usage
```python
from cobot_analytics import CobotDashboard

# Initialize dashboard
dashboard = CobotDashboard()

# Generate sample data
dashboard.generate_sample_data()

# Create safety summary
dashboard.create_safety_dashboard()

# Generate comprehensive report
report = dashboard.generate_safety_report()
```

### Custom Configuration
```python
# Customize safety thresholds
dashboard.sensor_data.proximity_threshold = 0.3  # meters
dashboard.sensor_data.force_threshold = 8.0      # Newtons
dashboard.sensor_data.safety_zone_radius = 1.2   # meters
```

## System Architecture

### Core Components

1. **CobotSensorData**: Simulates real-time sensor data
   - Proximity sensors
   - Force/torque sensors
   - Computer vision systems

2. **SafetyAnalyzer**: Analyzes safety patterns and metrics
   - Anomaly detection algorithms
   - Safety scoring calculations
   - Efficiency predictions

3. **CobotDashboard**: Main interface and reporting
   - Data aggregation
   - Report generation
   - Text-based summaries

### Data Flow
```
Sensor Data → Safety Analysis → Anomaly Detection → Report Generation
     ↓              ↓                    ↓              ↓
Proximity      Force Analysis    Pattern Recognition   Dashboard
Vision         Safety Scoring    Efficiency Metrics    Reports
```

## Configuration

### Safety Thresholds
- **Proximity Threshold**: Default 0.5m (minimum safe human distance)
- **Force Threshold**: Default 10.0N (maximum acceptable joint force)
- **Safety Zone Radius**: Default 1.0m (collaborative workspace boundary)

### Analysis Parameters
- **Contamination Rate**: 0.1 (10% expected anomalies)
- **Safety Score Weights**: Proximity (40%), Force (30%), Collision (30%)
- **Sample Size**: 1000 data points per hour (configurable)

## Output Examples

### Safety Dashboard Summary
```
PROXIMITY MONITORING:
• Average human distance: 1.87m
• Minimum distance: 0.12m
• Safety violations: 23

FORCE ANALYSIS:
• Joint 1 average force: 4.23N
• Joint 2 average force: 4.87N
• Joint 3 average force: 4.45N
• Excessive force events: 12
```

### Comprehensive Report
```
COBOT HUMAN-ROBOT INTERACTION ANALYTICS REPORT
================================================================
Overall Safety Score: 87.3%
• Proximity Safety: 92.1%
• Force Safety: 85.4%
• Collision Safety: 84.7%

COLLABORATION EFFICIENCY:
Overall Efficiency: 72.4%
• Collaboration Time Ratio: 68.2%
• Activity Diversity: 80.0%
• Force Smoothness: 69.1%
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Scikit-learn Not Available**
   - System automatically disables anomaly detection
   - Core functionality remains available

3. **Interactive Plot Issues**
   - System uses non-interactive matplotlib backend
   - All output is text-based for compatibility

### System Requirements
- **Memory**: ~50MB for standard operation
- **CPU**: Minimal processing requirements
- **Storage**: <1MB for script and dependencies

## Integration

### Industrial Systems
- Text output designed for log aggregation
- JSON-compatible data structures
- Configurable reporting intervals
- API-ready architecture

### Monitoring Integration
```python
# Example integration with monitoring system
dashboard = CobotDashboard()
dashboard.generate_sample_data()
report = dashboard.generate_safety_report()

# Extract key metrics
safety_score = report['safety_scores']['overall_safety']
efficiency = report['efficiency_metrics']['overall_efficiency']
anomalies = report['anomaly_count']
```

## Development

### Extending the System
1. **Custom Sensors**: Add new sensor types to `CobotSensorData`
2. **Analysis Algorithms**: Extend `SafetyAnalyzer` with custom metrics
3. **Reporting**: Modify `CobotDashboard` for custom output formats

### Contributing
- Follow PEP 8 style guidelines
- Add comprehensive error handling
- Include docstrings for new functions
- Test with various data scenarios

## License
- This project is designed for industrial and educational use. Modify and distribute according to your organization's requirements.

## Support
- For technical issues or feature requests:
1. Review the troubleshooting section
2. Check system requirements
3. Verify all dependencies are installed
4. Test with default configuration first

