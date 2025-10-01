"""
Training page for BuilderBrain Dashboard.

Detailed training monitoring, loss curves, and constraint evolution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import time


def render_training_page(data_loader, api_client):
    """Render the training monitoring page."""

    st.header("🎯 Training Monitor")

    # Training Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("▶️ Start Training", use_container_width=True):
            st.success("Training initiated!")
            st.rerun()

    with col2:
        if st.button("⏸️ Pause Training", use_container_width=True):
            st.warning("Training paused")
            st.rerun()

    with col3:
        st.info("Training controls would integrate with BuilderBrain trainer")

    # Load training data
    df = data_loader.get_training_metrics_df()
    training_progress = data_loader.get_training_progress_summary()
    latest_metrics = data_loader.get_latest_training_metrics()

    if df.empty:
        st.info("No training data available. Start training to see detailed metrics.")
        return

    # Training Progress Overview
    st.subheader("📈 Training Progress")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Step",
            latest_metrics.get('step', 0)
        )

    with col2:
        st.metric(
            "Current Loss",
            f"{latest_metrics.get('total_loss', 0):.4f}"
        )

    with col3:
        st.metric(
            "Epoch",
            latest_metrics.get('epoch', 0)
        )

    with col4:
        improvement_rate = training_progress.get('improvement_rate', 0)
        st.metric(
            "Improvement",
            f"{improvement_rate:.2%}",
            delta="↗️ Improving" if improvement_rate > 0.01 else "↘️ Declining"
        )

    # Loss Curves
    st.subheader("📊 Loss Evolution")

    # Main loss curves
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Loss Over Time', 'Task vs Total Loss', 'Constraint Losses', 'Dual Variables'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Total Loss
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['total_loss'], mode='lines+markers', name='Total Loss', line=dict(color='blue')),
        row=1, col=1
    )

    # Task Loss vs Total Loss
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['task_loss'], mode='lines+markers', name='Task Loss', line=dict(color='green')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['total_loss'], mode='lines+markers', name='Total Loss', line=dict(color='blue')),
        row=1, col=2, secondary_y=True
    )

    # Constraint Losses
    for col in df.columns:
        if col.startswith('constraint_'):
            constraint_name = col.replace('constraint_', '')
            fig.add_trace(
                go.Scatter(x=df['step'], y=df[col], mode='lines+markers', name=f'Constraint: {constraint_name}'),
                row=2, col=1
            )

    # Dual Variables
    for col in df.columns:
        if col.startswith('dual_'):
            dual_name = col.replace('dual_', '')
            fig.add_trace(
                go.Scatter(x=df['step'], y=df[col], mode='lines+markers', name=f'Dual: {dual_name}'),
                row=2, col=2
            )

    fig.update_layout(height=600, showlegend=True)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Dual Value", row=2, col=2)
    fig.update_xaxes(title_text="Training Step")

    st.plotly_chart(fig, use_container_width=True)

    # Constraint Analysis
    st.subheader("⚖️ Constraint Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Constraint violation heatmap
        constraint_cols = [col for col in df.columns if col.startswith('constraint_')]
        if constraint_cols:
            constraint_data = df[constraint_cols].iloc[-50:]  # Last 50 steps

            # Create correlation heatmap
            corr_matrix = constraint_data.corr()

            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title='Constraint Correlation Matrix',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Dual variable evolution
        dual_cols = [col for col in df.columns if col.startswith('dual_')]
        if dual_cols:
            dual_data = df[dual_cols].iloc[-50:]

            fig = go.Figure()
            for col in dual_cols:
                dual_name = col.replace('dual_', '')
                fig.add_trace(go.Scatter(
                    x=dual_data.index,
                    y=dual_data[col],
                    mode='lines+markers',
                    name=f'Dual: {dual_name}'
                ))

            fig.update_layout(
                title='Dual Variable Evolution',
                xaxis_title='Training Step',
                yaxis_title='Dual Value',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    # Training Statistics
    st.subheader("📊 Training Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Loss Statistics")
        if not df.empty:
            st.metric("Min Loss", f"{df['total_loss'].min():.4f}")
            st.metric("Max Loss", f"{df['total_loss'].max():.4f}")
            st.metric("Mean Loss", f"{df['total_loss'].mean():.4f}")
            st.metric("Std Loss", f"{df['total_loss'].std():.4f}")

    with col2:
        st.subheader("Constraint Stats")
        constraint_stats = {}
        for col in df.columns:
            if col.startswith('constraint_'):
                constraint_name = col.replace('constraint_', '')
                values = df[col].dropna()
                if len(values) > 0:
                    constraint_stats[constraint_name] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'violations': (values > 0.1).sum()
                    }

        for name, stats in constraint_stats.items():
            st.metric(f"{name.title()} Mean", f"{stats['mean']:.4f}")
            st.metric(f"{name.title()} Violations", stats['violations'])

    with col3:
        st.subheader("Training Health")
        # Calculate training health metrics
        recent_loss = df['total_loss'].tail(20)
        loss_stability = 1 - (recent_loss.std() / recent_loss.mean()) if recent_loss.mean() > 0 else 0

        st.metric("Loss Stability", f"{loss_stability:.2%}")

        # Convergence check
        early_avg = df['total_loss'].head(50).mean() if len(df) > 50 else df['total_loss'].mean()
        recent_avg = recent_loss.mean()
        convergence = (early_avg - recent_avg) / early_avg if early_avg > 0 else 0

        st.metric("Convergence Rate", f"{convergence:.2%}")

        # Constraint satisfaction
        total_violations = sum(
            (df[col] > 0.1).sum() for col in df.columns if col.startswith('constraint_')
        )
        total_constraints = sum(
            len(df[col].dropna()) for col in df.columns if col.startswith('constraint_')
        )
        constraint_satisfaction = 1 - (total_violations / total_constraints) if total_constraints > 0 else 1

        st.metric("Constraint Satisfaction", f"{constraint_satisfaction:.2%}")

    # Training History Table
    st.subheader("📋 Recent Training History")

    # Show last 10 steps
    recent_df = df.tail(10).copy()
    recent_df['step'] = recent_df['step'].astype(int)
    recent_df['epoch'] = recent_df['epoch'].astype(int)

    # Format numeric columns
    for col in recent_df.columns:
        if col not in ['step', 'epoch']:
            recent_df[col] = recent_df[col].round(4)

    st.dataframe(recent_df, use_container_width=True, height=300)

    # Export Options
    st.subheader("💾 Export Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Export Training Data (CSV)", use_container_width=True):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="training_history.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        if st.button("📈 Export Loss Curves (PNG)", use_container_width=True):
            # This would trigger a chart export in a real implementation
            st.info("Chart export functionality would be implemented here")
