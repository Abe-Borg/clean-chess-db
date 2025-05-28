# Chess Processing Performance Report

Generated: 20250527_194420

## System Information

- Platform: Windows-11-10.0.26100-SP0
- CPU: 8 physical cores, 16 logical cores
- Memory: 31.3 GB

## Performance Summary

**Best Configuration:** workers_8_adaptive
- Performance: 552.55 games/s
- Moves/s: 43014.82

## Identified Bottlenecks

### Synchronization Overhead
- **Description:** Efficiency decreases significantly with more workers
- **Impact:** Medium
- **Recommendation:** Optimize chunk size or reduce inter-process communication

## Recommendations

1. No significant performance issues detected. Consider profiling under different workloads or with larger datasets.
