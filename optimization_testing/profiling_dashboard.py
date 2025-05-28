# file: optimization_testing/profiling_dashboard.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
from typing import Dict, List, Any


class ProfilingDashboard:
    """Create interactive dashboard for profiling results."""
    
    def __init__(self, results_dir: str = "profiling_results"):
        self.results_dir = results_dir
        self.figures = []
        
    def create_dashboard(self, profiling_data: Dict[str, Any], 
                        output_file: str = "profiling_dashboard.html"):
        """Create interactive HTML dashboard from profiling results."""
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Performance Overview', 'Worker Efficiency', 'Memory Usage',
                'Component Timing', 'Scalability Analysis', 'CPU Utilization',
                'Bottleneck Analysis', 'Configuration Comparison', 'Final Metrics'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'sunburst'}, {'type': 'bar'}, {'type': 'indicator'}],
                [{'colspan': 3}, None, None]
            ],
            row_heights=[0.2, 0.3, 0.3, 0.2],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Performance Overview (Indicator)
        if 'final_metrics' in profiling_data:
            metrics = profiling_data['final_metrics']
            fig.add_trace(
                go.Indicator(
                    mode="number+gauge+delta",
                    value=metrics.get('average_games_per_second', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Games/Second"},
                    delta={'reference': 159, 'relative': True},
                    gauge={
                        'axis': {'range': [None, 1000]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 200], 'color': "lightgray"},
                            {'range': [200, 600], 'color': "gray"},
                            {'range': [600, 1000], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 784  # Your current performance
                        }
                    }
                ),
                row=1, col=1
            )
        
        # 2. Worker Efficiency
        if 'scalability' in profiling_data.get('validations', {}):
            worker_data = profiling_data['validations']['scalability']['worker_scaling']
            workers = sorted([int(w) for w in worker_data.keys()])
            efficiency = [worker_data[str(w)]['efficiency'] for w in workers]
            
            fig.add_trace(
                go.Bar(
                    x=workers,
                    y=efficiency,
                    name='Efficiency',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            fig.update_xaxes(title_text="Workers", row=1, col=2)
            fig.update_yaxes(title_text="Games/s/Worker", row=1, col=2)
        
        # 3. Memory Usage
        if 'memory' in profiling_data.get('profiling_data', {}):
            memory_data = profiling_data['profiling_data']['memory']['by_sample_size']
            sizes = sorted([int(s) for s in memory_data.keys()])
            memory_per_game = [memory_data[str(s)]['memory_per_game'] for s in sizes]
            
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=memory_per_game,
                    mode='lines+markers',
                    name='Memory/Game',
                    line=dict(color='red', width=2)
                ),
                row=1, col=3
            )
            fig.update_xaxes(title_text="Sample Size", row=1, col=3)
            fig.update_yaxes(title_text="MB per Game", row=1, col=3)
        
        # 4. Component Timing
        if 'components' in profiling_data.get('profiling_data', {}):
            components = profiling_data['profiling_data']['components']
            comp_names = list(components.keys())
            mean_times = [components[c]['mean_time'] * 1000 for c in comp_names]  # Convert to ms
            
            fig.add_trace(
                go.Bar(
                    x=comp_names,
                    y=mean_times,
                    name='Mean Time',
                    marker_color='green'
                ),
                row=2, col=1
            )
            fig.update_xaxes(title_text="Component", row=2, col=1)
            fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
        
        # 5. Scalability Analysis
        if 'scalability' in profiling_data.get('validations', {}):
            data_scaling = profiling_data['validations']['scalability']['data_scaling']
            sizes = sorted([int(s) for s in data_scaling.keys()])
            perf = [data_scaling[str(s)]['games_per_second'] for s in sizes]
            
            fig.add_trace(
                go.Scatter(
                    x=sizes,
                    y=perf,
                    mode='lines+markers',
                    name='Performance',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Dataset Size", type="log", row=2, col=2)
            fig.update_yaxes(title_text="Games/Second", row=2, col=2)
        
        # 6. CPU Utilization (placeholder for actual CPU data)
        if 'bottlenecks' in profiling_data:
            # Find CPU bottleneck data
            cpu_data = None
            for bottleneck in profiling_data['bottlenecks']:
                if bottleneck['type'] == 'CPU Underutilization':
                    cpu_data = bottleneck.get('metrics', {})
                    break
            
            if cpu_data:
                fig.add_trace(
                    go.Scatter(
                        x=['Average', 'Maximum'],
                        y=[cpu_data.get('avg_cpu', 0), cpu_data.get('max_cpu', 0)],
                        mode='markers+lines',
                        marker=dict(size=15),
                        name='CPU %'
                    ),
                    row=2, col=3
                )
                fig.update_yaxes(title_text="CPU %", range=[0, 100], row=2, col=3)
        
        # 7. Bottleneck Analysis (Sunburst)
        if 'bottlenecks' in profiling_data:
            bottleneck_data = self._prepare_bottleneck_sunburst(profiling_data['bottlenecks'])
            
            fig.add_trace(
                go.Sunburst(
                    labels=bottleneck_data['labels'],
                    parents=bottleneck_data['parents'],
                    values=bottleneck_data['values'],
                    branchvalues="total"
                ),
                row=3, col=1
            )
        
        # 8. Configuration Comparison
        if 'configurations' in profiling_data:
            configs = profiling_data['configurations']
            config_names = list(configs.keys())[:5]  # Top 5 configs
            games_per_sec = [configs[c]['games_per_second'] for c in config_names]
            
            fig.add_trace(
                go.Bar(
                    x=config_names,
                    y=games_per_sec,
                    name='Games/s',
                    marker_color='orange'
                ),
                row=3, col=2
            )
            fig.update_xaxes(tickangle=45, row=3, col=2)
            fig.update_yaxes(title_text="Games/Second", row=3, col=2)
        
        # 9. Final Metrics Summary
        if 'final_metrics' in profiling_data:
            metrics = profiling_data['final_metrics']
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=metrics.get('speedup_vs_baseline', 0),
                    title={'text': "Speedup vs Baseline"},
                    delta={'reference': 1, 'relative': False},
                    number={'suffix': "x"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=3, col=3
            )
        
        # 10. Performance History (bottom row)
        if 'performance_tests' in profiling_data:
            perf_tests = profiling_data['performance_tests']
            if perf_tests:
                # Create performance over time plot
                test_names = list(perf_tests.keys())
                performances = [perf_tests[t].get('games_per_second', 0) for t in test_names]
                
                fig.add_trace(
                    go.Scatter(
                        x=test_names,
                        y=performances,
                        mode='lines+markers',
                        name='Performance History',
                        line=dict(color='darkgreen', width=3),
                        fill='tozeroy'
                    ),
                    row=4, col=1
                )
                fig.update_xaxes(title_text="Test Configuration", row=4, col=1)
                fig.update_yaxes(title_text="Games/Second", row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title_text=f"Chess Processing Performance Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            showlegend=False,
            height=1500,
            template='plotly_white'
        )
        
        # Save to HTML
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"\nDashboard saved to: {output_file}")
        
        # Open in browser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        
        return fig
    
    def _prepare_bottleneck_sunburst(self, bottlenecks: List[Dict]) -> Dict:
        """Prepare data for bottleneck sunburst chart."""
        labels = ['Bottlenecks']
        parents = ['']
        values = [0]
        
        impact_values = {'High': 3, 'Medium': 2, 'Low': 1}
        
        for bottleneck in bottlenecks:
            labels.append(bottleneck['type'])
            parents.append('Bottlenecks')
            values.append(impact_values.get(bottleneck.get('impact', 'Low'), 1))
        
        return {'labels': labels, 'parents': parents, 'values': values}
    
    def create_comparison_dashboard(self, 
                                   results_list: List[Dict[str, Any]], 
                                   labels: List[str],
                                   output_file: str = "comparison_dashboard.html"):
        """Create dashboard comparing multiple profiling runs."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Performance Comparison', 'Memory Usage Comparison',
                'Scalability Comparison', 'Efficiency Trends'
            )
        )
        
        colors = px.colors.qualitative.Set1
        
        # 1. Performance Comparison
        performances = []
        for i, (result, label) in enumerate(zip(results_list, labels)):
            if 'final_metrics' in result:
                perf = result['final_metrics']['average_games_per_second']
                performances.append(perf)
                
                fig.add_trace(
                    go.Bar(
                        x=[label],
                        y=[perf],
                        name=label,
                        marker_color=colors[i % len(colors)]
                    ),
                    row=1, col=1
                )
        
        # 2. Memory Usage Comparison
        for i, (result, label) in enumerate(zip(results_list, labels)):
            if 'memory' in result.get('profiling_data', {}):
                memory_data = result['profiling_data']['memory']['by_sample_size']
                sizes = sorted([int(s) for s in memory_data.keys()])
                memory_per_game = [memory_data[str(s)]['memory_per_game'] for s in sizes]
                
                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=memory_per_game,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=1, col=2
                )
        
        # 3. Scalability Comparison
        for i, (result, label) in enumerate(zip(results_list, labels)):
            if 'scalability' in result.get('validations', {}):
                worker_data = result['validations']['scalability']['worker_scaling']
                workers = sorted([int(w) for w in worker_data.keys()])
                games_per_sec = [worker_data[str(w)]['games_per_second'] for w in workers]
                
                fig.add_trace(
                    go.Scatter(
                        x=workers,
                        y=games_per_sec,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=2, col=1
                )
        
        # 4. Efficiency Trends
        for i, (result, label) in enumerate(zip(results_list, labels)):
            if 'scalability' in result.get('validations', {}):
                worker_data = result['validations']['scalability']['worker_scaling']
                workers = sorted([int(w) for w in worker_data.keys()])
                efficiency = [worker_data[str(w)]['efficiency'] for w in workers]
                
                fig.add_trace(
                    go.Scatter(
                        x=workers,
                        y=efficiency,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Performance Comparison Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.update_xaxes(title_text="Configuration", row=1, col=1)
        fig.update_yaxes(title_text="Games/Second", row=1, col=1)
        fig.update_xaxes(title_text="Sample Size", row=1, col=2)
        fig.update_yaxes(title_text="Memory per Game (MB)", row=1, col=2)
        fig.update_xaxes(title_text="Number of Workers", row=2, col=1)
        fig.update_yaxes(title_text="Games/Second", row=2, col=1)
        fig.update_xaxes(title_text="Number of Workers", row=2, col=2)
        fig.update_yaxes(title_text="Efficiency (games/s/worker)", row=2, col=2)
        
        # Save to HTML
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"\nComparison dashboard saved to: {output_file}")
        
        # Open in browser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        
        return fig


def create_dashboard_from_results(results_path: str, output_file: str = None):
    """Create dashboard from saved profiling results."""
    
    # Load results
    with open(results_path, 'r') as f:
        profiling_data = json.load(f)
    
    # Create dashboard
    dashboard = ProfilingDashboard()
    
    if output_file is None:
        output_file = results_path.replace('.json', '_dashboard.html')
    
    dashboard.create_dashboard(profiling_data, output_file)
    
    return dashboard


def create_summary_report(profiling_data: Dict[str, Any], output_file: str = "summary_report.html"):
    """Create a text-based summary report with key findings."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chess Processing Performance Summary</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            .pass {{
                color: #28a745;
                font-weight: bold;
            }}
            .fail {{
                color: #dc3545;
                font-weight: bold;
            }}
            .recommendation {{
                background-color: #e9ecef;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .timestamp {{
                color: #666;
                font-size: 12px;
                text-align: right;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chess Processing Performance Summary</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Key Metrics
    if 'final_metrics' in profiling_data:
        metrics = profiling_data['final_metrics']
        html_content += """
            <h2>Key Performance Metrics</h2>
            <div>
        """
        
        html_content += f"""
                <div class="metric">
                    <div class="metric-value">{metrics.get('average_games_per_second', 0):.2f}</div>
                    <div class="metric-label">Games/Second</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('average_moves_per_second', 0):.0f}</div>
                    <div class="metric-label">Moves/Second</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('speedup_vs_baseline', 0):.2f}x</div>
                    <div class="metric-label">Speedup vs Baseline</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('average_corruption_rate', 0):.2f}%</div>
                    <div class="metric-label">Corruption Rate</div>
                </div>
            </div>
        """
    
    # Validation Results
    if 'validations' in profiling_data:
        html_content += """
            <h2>Validation Results</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
        """
        
        validations = profiling_data['validations']
        
        if 'false_corruption' in validations:
            status = "PASS" if validations['false_corruption']['passed'] else "FAIL"
            status_class = "pass" if validations['false_corruption']['passed'] else "fail"
            avg_corruption = validations['false_corruption']['average_corruption']
            html_content += f"""
                <tr>
                    <td>False Corruption Fix</td>
                    <td class="{status_class}">{status}</td>
                    <td>Average corruption rate: {avg_corruption:.2f}%</td>
                </tr>
            """
        
        if 'performance_targets' in validations:
            status = "PASS" if validations['performance_targets']['passed'] else "FAIL"
            status_class = "pass" if validations['performance_targets']['passed'] else "fail"
            avg_perf = validations['performance_targets']['average_games_per_second']
            html_content += f"""
                <tr>
                    <td>Performance Targets</td>
                    <td class="{status_class}">{status}</td>
                    <td>Average performance: {avg_perf:.2f} games/s</td>
                </tr>
            """
        
        html_content += "</table>"
    
    # Bottlenecks
    if 'bottlenecks' in profiling_data and profiling_data['bottlenecks']:
        html_content += """
            <h2>Identified Bottlenecks</h2>
        """
        
        for bottleneck in profiling_data['bottlenecks']:
            html_content += f"""
                <div class="recommendation">
                    <strong>{bottleneck['type']}</strong> (Impact: {bottleneck['impact']})<br>
                    {bottleneck['description']}<br>
                    <em>Recommendation: {bottleneck['recommendation']}</em>
                </div>
            """
    
    # Recommendations
    if 'recommendations' in profiling_data:
        html_content += """
            <h2>Optimization Recommendations</h2>
        """
        
        for i, rec in enumerate(profiling_data['recommendations'], 1):
            html_content += f"""
                <div class="recommendation">
                    {i}. {rec}
                </div>
            """
    
    # Configuration Comparison
    if 'configurations' in profiling_data:
        html_content += """
            <h2>Configuration Performance</h2>
            <table>
                <tr>
                    <th>Configuration</th>
                    <th>Games/Second</th>
                    <th>Moves/Second</th>
                    <th>Efficiency</th>
                </tr>
        """
        
        configs = profiling_data['configurations']
        sorted_configs = sorted(configs.items(), 
                               key=lambda x: x[1].get('games_per_second', 0), 
                               reverse=True)[:5]
        
        for config_name, config_data in sorted_configs:
            html_content += f"""
                <tr>
                    <td>{config_name}</td>
                    <td>{config_data.get('games_per_second', 0):.2f}</td>
                    <td>{config_data.get('moves_per_second', 0):.0f}</td>
                    <td>{config_data.get('efficiency', 0):.2f}</td>
                </tr>
            """
        
        html_content += "</table>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\nSummary report saved to: {output_file}")
    webbrowser.open(f'file://{os.path.abspath(output_file)}')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create profiling dashboard")
    parser.add_argument("--results", type=str, required=True, 
                       help="Path to profiling results JSON file")
    parser.add_argument("--output", type=str, help="Output HTML file path")
    parser.add_argument("--summary", action="store_true", 
                       help="Create summary report instead of dashboard")
    
    args = parser.parse_args()
    
    if args.summary:
        # Load results and create summary
        with open(args.results, 'r') as f:
            profiling_data = json.load(f)
        
        output_file = args.output or args.results.replace('.json', '_summary.html')
        create_summary_report(profiling_data, output_file)
    else:
        # Create interactive dashboard
        create_dashboard_from_results(args.results, args.output)