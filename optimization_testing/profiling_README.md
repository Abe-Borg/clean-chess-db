# Chess Game Processing Profiler

This profiling system helps identify performance bottlenecks and optimization opportunities in the chess game processing pipeline. The enhanced system provides comprehensive visualizations and reports to analyze performance metrics at different levels.

## Features

- **Function-level profiling**: Identify the most time-consuming functions
- **Memory usage analysis**: Track memory consumption patterns
- **CPU utilization monitoring**: Analyze CPU usage during processing
- **Worker performance tracking**: Monitor parallelism efficiency
- **Game-level metrics**: Identify which types of games or positions cause slowdowns
- **Cache efficiency analysis**: Analyze position cache hit/miss patterns
- **Comprehensive visualizations**: Generate dashboards, reports, and historical trends
- **Performance history tracking**: Monitor improvements over time

## Usage

The profiling script supports several modes and visualization options:

```bash
python profiling_script.py [--filepath FILEPATH] [--sample_size SAMPLE_SIZE] 
                           [--mode {profile,memory,cpu,all,game_level,worker,comprehensive}]
                           [--chunk_size CHUNK_SIZE] [--vis_dir VIS_DIR]
                           [--html_dashboard] [--pdf_report] [--track_history]
                           [--history_file HISTORY_FILE]
```

### Basic Options

- `--filepath`: Path to the DataFrame file to process (defaults to setting in game_settings)
- `--sample_size`: Number of games to sample for profiling (default: 5000, or 500 on Windows)
- `--mode`: Profiling mode to run:
  - `profile`: Basic cProfile analysis
  - `memory`: Memory usage scaling by sample size
  - `cpu`: CPU utilization monitoring
  - `game_level`: Game-level metrics analysis
  - `worker`: Worker performance monitoring
  - `comprehensive`: All profiling modes combined
  - `all`: Run profile, memory, and CPU modes
- `--chunk_size`: Chunk size for parallel processing (default: 100)

### Visualization Options

- `--vis_dir`: Output directory for visualizations (timestamped folder will be created)
- `--html_dashboard`: Generate interactive HTML dashboard
- `--pdf_report`: Generate comprehensive PDF report
- `--track_history`: Record performance to history file
- `--history_file`: CSV file for performance history (default: "performance_history.csv")

## Examples

### Basic Profile

```bash
python profiling_script.py --mode profile --sample_size 1000
```

### Comprehensive Analysis with Visualizations

```bash
python profiling_script.py --mode comprehensive --sample_size 2000 --vis_dir profiling_results --html_dashboard --pdf_report --track_history
```

### Worker Performance Analysis

```bash
python profiling_script.py --mode worker --sample_size 1000 --chunk_size 50 --html_dashboard
```

### Game-Level Performance Analysis

```bash
python profiling_script.py --mode game_level --sample_size 100 --pdf_report
```

### Track Performance History

```bash
python profiling_script.py --mode profile --sample_size 1000 --track_history --history_file performance_history.csv
```

## Visualization Examples

To generate example visualizations with sample data:

```bash
python visualization_examples.py --output_dir example_visualizations --type all
```

This will create:
- Interactive HTML dashboard
- Comprehensive PDF report
- Performance history trends

## Visualization Types

### HTML Dashboard

The interactive HTML dashboard provides a comprehensive overview of profiling results with:
- Performance metrics gauge
- CPU utilization chart
- Top functions by time
- Cache performance pie chart
- Performance recommendations

### PDF Report

The PDF report includes detailed visualizations for:
- Summary metrics
- CPU utilization over time
- Memory scaling
- Cache performance
- Top time-consuming functions
- Worker performance
- Game-level metrics
- Optimization recommendations

### Performance History

The performance history visualization tracks changes over time:
- Processing speed (games per second)
- Memory usage
- Cache efficiency

## Interpreting Results

### Key Metrics to Monitor

1. **Games per second**: Overall throughput of the system
2. **Memory usage trends**: Look for memory leaks or inefficient caching
3. **Cache hit rate**: Higher is better (>70% is good)
4. **Worker load imbalance**: Should be <10% for optimal parallelism
5. **CPU utilization**: Aim for >80% for maximum throughput
6. **Game length vs. processing time**: Look for anomalies that process slowly

### Common Optimization Opportunities

1. **Cache tuning**: If hit rate is low, consider preloading common positions
2. **Parallelism adjustment**: If worker load is imbalanced, adjust chunk size
3. **Time-heavy functions**: Optimize the functions consuming the most time
4. **Memory management**: If memory grows steadily, look for memory leaks
5. **Game-specific optimizations**: Identify what makes certain games slow

## Requirements

- Python 3.7+
- pandas
- matplotlib
- seaborn
- numpy
- psutil
- plotly (optional, for interactive dashboards)

## License

This profiling system is provided under the MIT License.