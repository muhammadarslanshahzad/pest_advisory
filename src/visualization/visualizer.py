"""
Visualization Module: Generate charts from encoder analysis
Uses matplotlib for high-quality, publication-ready charts
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, Any, List, Optional
import os

from src.config.config import (
    CHART_STYLE, CHART_DPI, CHART_COLORS, FIGURE_SIZES,
    CHARTS_OUTPUT_DIR, ETL_THRESHOLDS
)
from src.utils import sanitize_filename, get_timestamp



class PestVisualizer:
    """
    Creates publication-quality visualizations from encoder results
    """
    
    def __init__(self, output_dir: str = CHARTS_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Set seaborn style
        sns.set_theme()
        # plt.style.use(CHART_STYLE)
        # Set default DPI
        plt.rcParams['figure.dpi'] = CHART_DPI
        plt.rcParams['savefig.dpi'] = CHART_DPI
        
        # Set font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
        print(f"✅ Visualizer initialized - Output: {output_dir}")
    
    def create_pest_comparison_chart(self, context: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> str:
        """
        Chart 1: Pest counts vs ETL thresholds (Bar chart)
        
        Shows all pests with bars for actual counts and line for ETL
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['pest_bar'])
        
        # Extract pest data
        pest_data = context['pest_data']
        
        # Filter to show only non-zero or above-threshold pests
        pests_to_show = {
            name: data for name, data in pest_data.items()
            if data['count'] > 0 or data['status'] == 'ABOVE'
        }
        
        if not pests_to_show:
            # If all zeros, show top 5 common pests anyway
            common_pests = ['Whitefly', 'Jassid', 'Pink Bollworm', 
                          'American Bollworm', 'Thrips']
            pests_to_show = {
                name: pest_data[name] 
                for name in common_pests 
                if name in pest_data
            }
        
        # Prepare data
        pest_names = list(pests_to_show.keys())
        actual_counts = [pests_to_show[p]['count'] for p in pest_names]
        etl_thresholds = [pests_to_show[p]['threshold'] for p in pest_names]
        statuses = [pests_to_show[p]['status'] for p in pest_names]
        
        # Color bars based on status
        bar_colors = [
            CHART_COLORS['ABOVE_ETL'] if s == 'ABOVE' else CHART_COLORS['BELOW_ETL']
            for s in statuses
        ]
        
        # Create bars
        x_pos = np.arange(len(pest_names))
        bars = ax.bar(x_pos, actual_counts, color=bar_colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.2, label='Actual Count')
        
        # Add ETL threshold line
        ax.plot(x_pos, etl_thresholds, 'ko--', linewidth=2, 
               markersize=8, label='ETL Threshold', zorder=10)
        
        # Add value labels on bars
        for i, (bar, count, threshold, status) in enumerate(
            zip(bars, actual_counts, etl_thresholds, statuses)
        ):
            height = bar.get_height()
            
            # Label above bar
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Show percentage above ETL if applicable
            if status == 'ABOVE' and threshold > 0:
                pct_above = ((count - threshold) / threshold) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'+{pct_above:.0f}%',
                       ha='center', va='center', fontsize=9,
                       color='white', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
        
        # Styling
        ax.set_xlabel('Pest Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Count per Spot', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Pest Population vs ETL Thresholds\n{context["tehsil"]} - {context["timeframe"]}',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pest_names, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add severity badge
        severity_color = CHART_COLORS[context['severity_level']]
        ax.text(0.02, 0.98, f"Severity: {context['severity_level']}",
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=severity_color, alpha=0.7))
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            filename = f"pest_comparison_{sanitize_filename(context['tehsil'])}_{get_timestamp()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
        return save_path
    
    def create_severity_gauge(self, context: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
        """
        Chart 2: Severity gauge showing risk level
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['severity_gauge'])
        
        severity_score = context['severity_score']
        severity_level = context['severity_level']
        
        # Define zones
        zones = [
            (0, 5, CHART_COLORS['LOW'], 'LOW'),
            (5, 15, CHART_COLORS['MEDIUM'], 'MEDIUM'),
            (15, 30, CHART_COLORS['HIGH'], 'HIGH')
        ]
        
        # Draw colored bands
        for start, end, color, label in zones:
            width = end - start
            rect = mpatches.Rectangle((start, 0), width, 0.4, 
                                     facecolor=color, alpha=0.6,
                                     edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add zone label
            ax.text(start + width/2, 0.5, label,
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Draw pointer (triangle)
        pointer_x = min(severity_score, 30)  # Cap at 30
        triangle = mpatches.Polygon(
            [[pointer_x - 0.5, -0.05], [pointer_x + 0.5, -0.05], [pointer_x, 0.2]],
            closed=True, facecolor='black', edgecolor='black', linewidth=2
        )
        ax.add_patch(triangle)
        
        # Add score label
        ax.text(pointer_x, -0.2, f'{severity_score:.1f}',
               ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        
        # Styling
        ax.set_xlim(-1, 31)
        ax.set_ylim(-0.3, 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_title(
            f'Overall Threat Level: {severity_level}\n{context["tehsil"]} - {context["timeframe"]}',
            fontsize=14, fontweight='bold', pad=10
        )
        
        # Add tick marks
        for tick in [0, 5, 10, 15, 20, 25, 30]:
            ax.plot([tick, tick], [-0.02, 0.02], 'k-', linewidth=1)
            ax.text(tick, -0.08, str(tick), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            filename = f"severity_gauge_{sanitize_filename(context['tehsil'])}_{get_timestamp()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
        return save_path
    
    def create_threat_pests_chart(self, context: Dict[str, Any],
                                  save_path: Optional[str] = None) -> str:
        """
        Chart 3: Horizontal bar chart of critical threat pests
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['threat_horizontal'])
        
        threat_pests = context.get('threat_pests', [])
        
        if not threat_pests:
            # No threats - show message
            ax.text(0.5, 0.5, '✓ No Critical Threats Detected',
                   ha='center', va='center', fontsize=16, 
                   fontweight='bold', color='green',
                   transform=ax.transAxes)
            ax.text(0.5, 0.4, 'All pest populations below critical levels',
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, style='italic')
            ax.axis('off')
        else:
            # Extract data
            pest_names = [p['name'] for p in threat_pests]
            multipliers = [p['multiplier'] for p in threat_pests]
            severities = [p['severity'] for p in threat_pests]
            
            # Colors based on severity
            bar_colors = [
                CHART_COLORS['CRITICAL'] if s == 'CRITICAL' else CHART_COLORS['HIGH']
                for s in severities
            ]
            
            # Create horizontal bars
            y_pos = np.arange(len(pest_names))
            bars = ax.barh(y_pos, multipliers, color=bar_colors, alpha=0.8,
                          edgecolor='black', linewidth=1.2)
            
            # Add ETL line at x=1
            ax.axvline(x=1, color='black', linestyle='--', linewidth=2,
                      label='ETL Threshold', zorder=0)
            
            # Add critical line at x=5
            ax.axvline(x=5, color='red', linestyle=':', linewidth=2,
                      label='Critical (5x ETL)', zorder=0)
            
            # Add value labels
            for i, (bar, mult, pest) in enumerate(zip(bars, multipliers, threat_pests)):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                       f'{mult:.1f}x  ({pest["count"]:.1f} count)',
                       va='center', fontsize=10, fontweight='bold')
            
            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pest_names, fontsize=11)
            ax.set_xlabel('Times Above ETL Threshold', fontsize=12, fontweight='bold')
            ax.set_title(
                f'Critical Threat Pests\n{context["tehsil"]} - {context["timeframe"]}',
                fontsize=14, fontweight='bold', pad=20
            )
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Set x-axis limit
            max_multiplier = max(multipliers)
            ax.set_xlim(0, max_multiplier * 1.2)
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            filename = f"threat_pests_{sanitize_filename(context['tehsil'])}_{get_timestamp()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
        return save_path
    
    def create_disease_chart(self, context: Dict[str, Any],
                            save_path: Optional[str] = None) -> str:
        """
        Chart 4: Disease infection rates (CLCuV and Wilt)
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['disease_bar'])
        
        disease_data = context.get('disease_data', {})
        
        if not disease_data:
            ax.text(0.5, 0.5, 'No Disease Data Available',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)
            ax.axis('off')
        else:
            # Prepare data
            diseases = []
            spot_infections = []
            area_infections = []
            
            for disease_name, data in disease_data.items():
                diseases.append(disease_name)
                spot_infections.append(data['spot_infection'])
                area_infections.append(data['area_infection'])
            
            # Create grouped bar chart
            x_pos = np.arange(len(diseases))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, spot_infections, width,
                          label='Spot Infection %', color='#FF6B6B', alpha=0.8,
                          edgecolor='black', linewidth=1)
            bars2 = ax.bar(x_pos + width/2, area_infections, width,
                          label='Area Infection %', color='#4ECDC4', alpha=0.8,
                          edgecolor='black', linewidth=1)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.1f}%',
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add danger threshold line at 30%
            ax.axhline(y=30, color='red', linestyle='--', linewidth=2,
                      alpha=0.5, label='High Risk (30%)')
            
            # Styling
            ax.set_xlabel('Disease Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Infection Rate (%)', fontsize=12, fontweight='bold')
            ax.set_title(
                f'Disease Infection Rates\n{context["tehsil"]} - {context["timeframe"]}',
                fontsize=14, fontweight='bold', pad=20
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(diseases, fontsize=11)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.set_ylim(0, max(max(spot_infections + area_infections, default=10), 40))
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            filename = f"disease_rates_{sanitize_filename(context['tehsil'])}_{get_timestamp()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
        return save_path
    
    def create_summary_dashboard(self, context: Dict[str, Any], 
                                encoder_analysis: Dict[str, Any],
                                save_path: Optional[str] = None) -> str:
        """
        Chart 5: Comprehensive dashboard with all key metrics
        """
        fig = plt.figure(figsize=FIGURE_SIZES['summary_grid'])
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Title
        fig.suptitle(
            f'Pest Advisory Dashboard: {context["tehsil"]}\n{context["timeframe"]}',
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # --- Panel 1: Key Metrics (Top Left) ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        metrics_text = f"""
SURVEY COVERAGE
- Spots Surveyed: {context['total_spots']}
- Total Area: {context['total_area']:.1f} acres
- Pest Pressure: {context['pest_pressure_per_acre']:.2f}/acre

RISK ASSESSMENT
- Severity Level: {context['severity_level']}
- Severity Score: {context['severity_score']:.1f}/30
- Risk Level: {encoder_analysis['risk_level']}
- Action Urgency: {encoder_analysis['action_urgency']}
- Economic Impact: {encoder_analysis['economic_impact']}

CRITICAL THREATS
"""
        if context['threat_pests']:
            for pest in context['threat_pests'][:3]:
                metrics_text += f"• {pest['name']}: {pest['multiplier']:.1f}x ETL ({pest['severity']})\n"
        else:
            metrics_text += "• None detected\n"
        
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        # --- Panel 2: Severity Gauge (Top Right) ---
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_mini_gauge(ax2, context)
        
        # --- Panel 3: Pest Comparison (Middle) ---
        ax3 = fig.add_subplot(gs[1, :])
        self._draw_mini_pest_bars(ax3, context)
        
        # --- Panel 4: Disease Status (Bottom Left) ---
        ax4 = fig.add_subplot(gs[2, 0])
        self._draw_mini_disease(ax4, context)
        
        # --- Panel 5: Key Insights (Bottom Right) ---
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        insights_text = "KEY INSIGHTS\n\n"
        for i, insight in enumerate(encoder_analysis.get('key_insights', [])[:4], 1):
            insights_text += f"{i}. {insight}\n\n"
        
        ax5.text(0.05, 0.95, insights_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Save
        if save_path is None:
            filename = f"dashboard_{sanitize_filename(context['tehsil'])}_{get_timestamp()}.png"
            save_path = os.path.join(self.output_dir, filename)
        
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Saved: {save_path}")
        return save_path
    
    # Helper methods for dashboard panels
    def _draw_mini_gauge(self, ax, context):
        """Draw compact severity gauge"""
        severity_score = context['severity_score']
        severity_level = context['severity_level']
        
        zones = [(0, 5, CHART_COLORS['LOW']), (5, 15, CHART_COLORS['MEDIUM']), 
                (15, 30, CHART_COLORS['HIGH'])]
        
        for start, end, color in zones:
            ax.barh(0, end-start, left=start, height=0.3, color=color, alpha=0.6)
        
        ax.plot([severity_score], [0], marker='v', markersize=15, color='black')
        ax.text(severity_score, -0.15, f'{severity_score:.1f}', 
               ha='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 30)
        ax.set_ylim(-0.3, 0.3)
        ax.set_yticks([])
        ax.set_title(f'Severity: {severity_level}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    def _draw_mini_pest_bars(self, ax, context):
        """Draw compact pest comparison"""
        pest_data = context['pest_data']
        top_pests = sorted(
            [(name, data) for name, data in pest_data.items()],
            key=lambda x: x[1]['count'],
            reverse=True
        )[:6]
        
        names = [p[0] for p in top_pests]
        counts = [p[1]['count'] for p in top_pests]
        thresholds = [p[1]['threshold'] for p in top_pests]
        colors = [CHART_COLORS['ABOVE_ETL'] if p[1]['status'] == 'ABOVE' 
                 else CHART_COLORS['BELOW_ETL'] for p in top_pests]
        
        x_pos = np.arange(len(names))
        ax.bar(x_pos, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.plot(x_pos, thresholds, 'ko--', linewidth=2, label='ETL')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Top Pests vs ETL', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    def _draw_mini_disease(self, ax, context):
        """Draw compact disease chart"""
        disease_data = context.get('disease_data', {})
        
        if not disease_data or all(d['spot_infection'] == 0 and d['area_infection'] == 0 
                                  for d in disease_data.values()):
            ax.text(0.5, 0.5, '✓ No Disease Detected', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='green', transform=ax.transAxes)
            ax.axis('off')
        else:
            diseases = list(disease_data.keys())
            spot_inf = [disease_data[d]['spot_infection'] for d in diseases]
            area_inf = [disease_data[d]['area_infection'] for d in diseases]
            
            x_pos = np.arange(len(diseases))
            width = 0.35
            
            ax.bar(x_pos - width/2, spot_inf, width, label='Spot %', color='#FF6B6B', alpha=0.8)
            ax.bar(x_pos + width/2, area_inf, width, label='Area %', color='#4ECDC4', alpha=0.8)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(diseases, fontsize=10)
            ax.set_ylabel('Infection %', fontsize=10)
            ax.set_title('Disease Status', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
    
    def generate_all_charts(self, encoder_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all charts for a given encoder result
        
        Returns: Dictionary mapping chart names to file paths
        """
        context = encoder_result['context']
        analysis = encoder_result['encoder_analysis']
        
        print(f"\n📊 Generating visualizations for {context['tehsil']}...")
        
        charts = {}
        
        try:
            charts['pest_comparison'] = self.create_pest_comparison_chart(context)
        except Exception as e:
            print(f"   ⚠️  Pest comparison failed: {e}")
        
        try:
            charts['severity_gauge'] = self.create_severity_gauge(context)
        except Exception as e:
            print(f"   ⚠️  Severity gauge failed: {e}")
        
        try:
            charts['threat_pests'] = self.create_threat_pests_chart(context)
        except Exception as e:
            print(f"   ⚠️  Threat pests failed: {e}")
        
        try:
            charts['disease'] = self.create_disease_chart(context)
        except Exception as e:
            print(f"   ⚠️  Disease chart failed: {e}")
        
        try:
            charts['dashboard'] = self.create_summary_dashboard(context, analysis)
        except Exception as e:
            print(f"   ⚠️  Dashboard failed: {e}")
        
        print(f"✅ Generated {len(charts)} charts")
        
        return charts

    def generate_batch(self, encoder_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Generate charts for multiple encoder results in one pass.
        
        Args:
            encoder_results: List of encoder results produced by PestDataEncoder.
        
        Returns:
            Mapping of tehsil identifier → chart name → file path.
        """
        batch_outputs: Dict[str, Dict[str, str]] = {}
        
        if not encoder_results:
            print("⚠️  No encoder results provided for batch visualization.")
            return batch_outputs
        
        print(f"\n🗺️  Starting batch visualization for {len(encoder_results)} tehsil(s)...")
        
        for idx, encoder_result in enumerate(encoder_results, start=1):
            context = encoder_result.get('context', {})
            tehsil_name = context.get('tehsil', f"tehsil_{idx}")
            tehsil_key = sanitize_filename(tehsil_name)
            
            print(f"\n[{idx}/{len(encoder_results)}] 📍 {tehsil_name}")
            
            try:
                charts = self.generate_all_charts(encoder_result)
                batch_outputs[tehsil_key] = charts
            except Exception as exc:
                print(f"   ❌ Failed to generate charts for {tehsil_name}: {exc}")
        
        print(f"\n✅ Batch visualization complete for {len(batch_outputs)} tehsil(s)")
        return batch_outputs


# Test function
def test_visualizer():
    """Test visualizer with sample encoder result"""
    import json
    
    # Load a sample result
    sample_file = "outputs/encoder_results/muzufar_garh_1_week_september_2020_20251030_033526.json"
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        print("Run encoder first to generate sample data")
        return
    
    with open(sample_file, 'r') as f:
        encoder_result = json.load(f)
    
    # Initialize visualizer
    visualizer = PestVisualizer()
    
    # Generate all charts
    charts = visualizer.generate_all_charts(encoder_result)
    
    print("\n✅ Visualization test complete!")
    print(f"Charts saved to: {CHARTS_OUTPUT_DIR}")
    
    return charts


if __name__ == "__main__":
    test_visualizer()
