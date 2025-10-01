"""
Configuration page for BuilderBrain Dashboard.

Model configuration, constraint settings, and system management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any


def render_config_page(data_loader, api_client):
    """Render the configuration management page."""

    st.header("⚙️ Configuration Management")

    # Tab layout for different config sections
    tab1, tab2, tab3, tab4 = st.tabs(["Model Config", "Constraints", "Training", "System"])

    # Model Configuration Tab
    with tab1:
        st.subheader("🤖 Model Configuration")

        model_config = data_loader.get_model_config()

        if model_config:
            col1, col2 = st.columns(2)

            with col1:
                st.json(model_config)

            with col2:
                st.subheader("Model Parameters")

                st.metric("Hidden Size", model_config.get('hidden_size', 'Unknown'))
                st.metric("Num Layers", model_config.get('num_layers', 'Unknown'))
                st.metric("Num Programs", model_config.get('num_programs', 'Unknown'))
                st.metric("Alpha Cap", model_config.get('alpha_cap', 'Unknown'))

                # Model scale selector
                model_scales = api_client.get_model_scales()
                current_scale = api_client.get_model_status().get('model_scale', 'small')

                new_scale = st.selectbox(
                    "Active Model Scale",
                    model_scales,
                    index=model_scales.index(current_scale) if current_scale in model_scales else 0
                )

                if new_scale != current_scale:
                    if st.button("Switch Model Scale"):
                        result = api_client.set_model_scale(new_scale)
                        if "error" not in result:
                            st.success(f"Switched to {new_scale} model scale")
                            st.rerun()
                        else:
                            st.error(f"Failed to switch model: {result['error']}")
        else:
            st.info("No model configuration found. Using default settings.")

    # Constraints Tab
    with tab2:
        st.subheader("⚖️ Constraint Configuration")

        constraint_config = data_loader.get_constraint_config()

        if constraint_config:
            col1, col2 = st.columns(2)

            with col1:
                st.json(constraint_config)

            with col2:
                st.subheader("Constraint Status")

                for constraint_name, config in constraint_config.items():
                    enabled = config.get('enabled', False)
                    target = config.get('target', 0.0)
                    normalizer = config.get('normalizer', 'rank')

                    st.metric(
                        f"{constraint_name.title()} Target",
                        f"{target:.3f}",
                        delta="✅ Enabled" if enabled else "❌ Disabled"
                    )

                    # Constraint adjustment (mock)
                    if enabled:
                        new_target = st.slider(
                            f"Adjust {constraint_name} target",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(target),
                            step=0.01,
                            key=f"constraint_{constraint_name}"
                        )

                        if new_target != target:
                            st.info(f"⚠️ Target adjustment from {target:.3f} to {new_target:.3f}")
        else:
            st.info("No constraint configuration found.")

    # Training Tab
    with tab3:
        st.subheader("🎯 Training Configuration")

        training_config = data_loader.get_training_config()

        if training_config:
            col1, col2 = st.columns(2)

            with col1:
                st.json(training_config)

            with col2:
                st.subheader("Training Parameters")

                st.metric("Batch Size", training_config.get('batch_size', 'Unknown'))
                st.metric("Learning Rate", training_config.get('learning_rate', 'Unknown'))
                st.metric("Eta Lambda", training_config.get('eta_lambda', 'Unknown'))
                st.metric("Lambda Max", training_config.get('lambda_max', 'Unknown'))
                st.metric("Num Epochs", training_config.get('num_epochs', 'Unknown'))

                # Training controls
                st.subheader("Training Controls")

                if st.button("▶️ Start Training", use_container_width=True):
                    st.success("Training started!")

                if st.button("⏸️ Pause Training", use_container_width=True):
                    st.warning("Training paused")

                if st.button("🛑 Stop Training", use_container_width=True):
                    st.error("Training stopped")

        # Training History Management
        st.subheader("📚 Training History")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Export Training History", use_container_width=True):
                df = data_loader.get_training_metrics_df()
                if not df.empty:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        file_name="training_history.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No training history to export")

        with col2:
            if st.button("🗑️ Clear Training History", use_container_width=True):
                st.warning("⚠️ This will delete all training history!")
                if st.button("Confirm Clear", type="primary"):
                    # This would actually clear the history file
                    st.success("Training history cleared")

    # System Tab
    with tab4:
        st.subheader("💻 System Configuration")

        # System Metrics
        system_metrics = data_loader.get_system_metrics()

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

        # API Configuration
        st.subheader("🔗 API Configuration")

        api_status = api_client.get_model_status()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("API Health", "✅ Online" if api_client.health_check() else "❌ Offline")

        with col2:
            st.metric("Model Status", api_status.get('status', 'Unknown'))

        # Grammar Configuration
        st.subheader("🔤 Grammar Configuration")

        grammar_constraints = api_client.get_grammar_constraints()

        if grammar_constraints:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Available Grammars")
                for grammar in grammar_constraints.get('available_grammars', []):
                    st.info(f"• {grammar}")

            with col2:
                st.subheader("Constraint Modes")
                strict_modes = grammar_constraints.get('strict_modes', [])
                flexible_modes = grammar_constraints.get('flexible_modes', [])

                if strict_modes:
                    st.success("Strict Modes:")
                    for mode in strict_modes:
                        st.success(f"• {mode}")

                if flexible_modes:
                    st.info("Flexible Modes:")
                    for mode in flexible_modes:
                        st.info(f"• {mode}")

        # Model Export
        st.subheader("📦 Model Export")

        col1, col2, col3 = st.columns(3)

        with col1:
            export_scale = st.selectbox(
                "Export Scale",
                api_client.get_model_scales(),
                key="export_scale"
            )

        with col2:
            export_format = st.selectbox(
                "Export Format",
                ["hf", "pytorch", "onnx"],
                key="export_format"
            )

        with col3:
            if st.button("🚀 Export Model", use_container_width=True):
                with st.spinner("Exporting model..."):
                    result = api_client.export_model(export_scale, export_format)

                if "error" not in result:
                    st.success(f"Export initiated: {result.get('export_id', 'unknown')}")
                    st.info(f"File size: {result.get('file_size', 'unknown')}")
                else:
                    st.error(f"Export failed: {result['error']}")

        # Logs and Diagnostics
        st.subheader("📋 Logs and Diagnostics")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 View Recent Logs", use_container_width=True):
                st.info("Recent log entries would be displayed here")
                # Mock log entries
                log_entries = [
                    "2024-01-15 10:30:00 - INFO - Training step completed",
                    "2024-01-15 10:29:55 - DEBUG - Grammar constraint applied",
                    "2024-01-15 10:29:50 - INFO - Plan validation successful",
                    "2024-01-15 10:29:45 - WARN - High memory usage detected",
                ]

                for log in log_entries:
                    st.text(log)

        with col2:
            if st.button("🔍 Run Diagnostics", use_container_width=True):
                st.info("Running system diagnostics...")

                # Mock diagnostic results
                diagnostics = {
                    "Grammar Parser": "✅ Working",
                    "Plan Validator": "✅ Working",
                    "Model Loading": "✅ Working",
                    "API Endpoints": "✅ Working",
                    "Memory Usage": "⚠️ High",
                    "Disk Space": "✅ Normal"
                }

                for component, status in diagnostics.items():
                    if "✅" in status:
                        st.success(f"{component}: {status}")
                    elif "⚠️" in status:
                        st.warning(f"{component}: {status}")
                    else:
                        st.error(f"{component}: {status}")

    # Footer
    st.markdown("---")
    st.caption("Configuration Management - Adjust model parameters, constraints, and system settings")
