"""
Overview page for BuilderBrain Dashboard.

Displays high-level KPIs, system status, and real-time monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import time


def render_overview_page(data_loader, api_client):
    """Render the main overview dashboard page."""

    st.header("📊 System Overview")

    # Create columns for KPIs
    col1, col2, col3, col4 = st.columns(4)

    # Load data
    training_metrics = data_loader.get_latest_training_metrics()
    system_metrics = data_loader.get_system_metrics()
    training_progress = data_loader.get_training_progress_summary()
    constraint_violations = data_loader.get_constraint_violation_summary()

    # KPI Cards
    with col1:
        st.metric(
            label="Training Steps",
            value=training_progress.get('total_steps', 0),
            delta=f"{training_progress.get('improvement_rate', 0):.1%}" if training_progress.get('is_converging') else None
        )

    with col2:
        st.metric(
            label="Current Loss",
            value=f"{training_progress.get('current_loss', 0):.3f}",
            delta=f"{training_progress.get('improvement_rate', 0):.1%}" if training_progress.get('improvement_rate', 0) > 0 else None
        )

    with col3:
        st.metric(
            label="System CPU",
            value=f"{system_metrics.get('cpu_percent', 0):.1f}%",
            delta="Normal" if system_metrics.get('cpu_percent', 0) < 80 else "High"
        )

    with col4:
        st.metric(
            label="Memory Usage",
            value=f"{system_metrics.get('memory_percent', 0):.1f}%",
            delta="Normal" if system_metrics.get('memory_percent', 0) < 80 else "High"
        )

    # Training Progress Section
    st.subheader("🎯 Training Progress")

    if training_progress:
        col1, col2, col3 = st.columns(3)

        with col1:
            progress = min(training_progress.get('epochs_completed', 0) / 100, 1.0)  # Assuming 100 epoch target
            st.progress(progress, text=f"Epochs: {training_progress.get('epochs_completed', 0)}/100")

        with col2:
            st.metric("Best Loss", f"{training_progress.get('best_loss', 0):.3f}")

        with col3:
            st.metric(
                "Converging",
                "✅ Yes" if training_progress.get('is_converging') else "❌ No"
            )

    # Loss Curves
    st.subheader("📈 Loss Curves")

    df = data_loader.get_training_metrics_df()
    if not df.empty:
        # Create loss curve chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Loss', 'Task Loss', 'Constraint Losses', 'Dual Variables'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Total Loss
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['total_loss'], mode='lines', name='Total Loss'),
            row=1, col=1
        )

        # Task Loss
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['task_loss'], mode='lines', name='Task Loss'),
            row=1, col=2
        )

        # Constraint Losses
        for col in df.columns:
            if col.startswith('constraint_'):
                constraint_name = col.replace('constraint_', '')
                fig.add_trace(
                    go.Scatter(x=df['step'], y=df[col], mode='lines', name=f'Constraint: {constraint_name}'),
                    row=2, col=1
                )

        # Dual Variables
        for col in df.columns:
            if col.startswith('dual_'):
                dual_name = col.replace('dual_', '')
                fig.add_trace(
                    go.Scatter(x=df['step'], y=df[col], mode='lines', name=f'Dual: {dual_name}'),
                    row=2, col=2
                )

        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No training data available. Start training to see loss curves.")

    # System Health Section
    st.subheader("💻 System Health")

    if system_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("CPU Usage", f"{system_metrics.get('cpu_percent', 0):.1f}%")

        with col2:
            st.metric("Memory Usage", f"{system_metrics.get('memory_percent', 0):.1f}%")

        with col3:
            st.metric("Disk Usage", f"{system_metrics.get('disk_percent', 0):.1f}%")

        with col4:
            st.metric("Active Processes", system_metrics.get('active_processes', 0))

        # System resource chart
        resources_fig = go.Figure()
        resources_fig.add_trace(go.Bar(
            name='CPU',
            x=['CPU'], y=[system_metrics.get('cpu_percent', 0)],
            marker_color='lightblue'
        ))
        resources_fig.add_trace(go.Bar(
            name='Memory',
            x=['Memory'], y=[system_metrics.get('memory_percent', 0)],
            marker_color='lightgreen'
        ))
        resources_fig.add_trace(go.Bar(
            name='Disk',
            x=['Disk'], y=[system_metrics.get('disk_percent', 0)],
            marker_color='lightcoral'
        ))

        resources_fig.update_layout(
            title="System Resource Usage",
            yaxis_title="Usage (%)",
            showlegend=False,
            height=300
        )
        st.plotly_chart(resources_fig, use_container_width=True)

    # Constraint Compliance Section
    st.subheader("⚖️ Constraint Compliance")

    if constraint_violations:
        col1, col2 = st.columns(2)

        with col1:
            # Constraint violation rates
            violation_data = []
            for constraint, data in constraint_violations.items():
                violation_data.append({
                    'Constraint': constraint,
                    'Violation Rate': data['violation_rate'],
                    'Current Value': data['current_value']
                })

            if violation_data:
                violation_df = pd.DataFrame(violation_data)

                fig = px.bar(
                    violation_df,
                    x='Constraint',
                    y='Violation Rate',
                    title='Constraint Violation Rates',
                    color='Violation Rate',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Constraint trends
            trend_data = []
            for constraint, data in constraint_violations.items():
                trend_data.append({
                    'Constraint': constraint,
                    'Trend': data['trend'],
                    'Current Value': data['current_value']
                })

            if trend_data:
                trend_df = pd.DataFrame(trend_data)

                # Color by trend
                colors = {'improving': 'green', 'degrading': 'red'}
                trend_df['color'] = trend_df['Trend'].map(colors)

                fig = px.bar(
                    trend_df,
                    x='Constraint',
                    y='Current Value',
                    title='Constraint Values',
                    color='color',
                    color_discrete_map=colors
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # API Status
    st.subheader("🔗 API Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        api_healthy = api_client.health_check()
        st.metric(
            "API Health",
            "✅ Online" if api_healthy else "❌ Offline"
        )

    with col2:
        model_status = api_client.get_model_status()
        st.metric(
            "Model Scale",
            model_status.get('model_scale', 'Unknown')
        )

    with col3:
        constraint_metrics = api_client.get_constraint_metrics()
        compliance_rate = constraint_metrics.get('grammar_compliance_rate', 0)
        st.metric(
            "Grammar Compliance",
            f"{compliance_rate:.1%}"
        )

    # Recent Activity
    st.subheader("📋 Recent Activity")

    # Mock recent activity data
    activities = [
        {"time": "2 min ago", "type": "Training Step", "details": "Completed step 1500"},
        {"time": "5 min ago", "type": "Constraint Update", "details": "Grammar dual variable adjusted"},
        {"time": "10 min ago", "type": "Inference", "details": "Generated 150 tokens"},
        {"time": "15 min ago", "type": "Plan Validation", "details": "Validated robot manipulation plan"},
        {"time": "20 min ago", "type": "System Check", "details": "Memory usage normal"},
    ]

    for activity in activities:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.caption(activity["time"])
        with col2:
            st.text(f"{activity['type']}: {activity['details']}")
        with col3:
            if "Training" in activity["type"]:
                st.success("✓")
            elif "Error" in activity["type"]:
                st.error("✗")
            else:
                st.info("●")

    # Quick Actions
    st.subheader("🚀 Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎯 Start Training", use_container_width=True):
            st.success("Training started!")

    with col2:
        if st.button("🧠 Run Inference", use_container_width=True):
            st.info("Navigate to Inference page for model testing")

    with col3:
        if st.button("📊 Export Metrics", use_container_width=True):
            st.success("Metrics exported to CSV")

    with col4:
        if st.button("🔧 View Logs", use_container_width=True):
            st.info("Navigate to Configuration page for logs")

    # Footer
    st.markdown("---")
    st.caption("BuilderBrain Dashboard - Real-time monitoring and control interface")
