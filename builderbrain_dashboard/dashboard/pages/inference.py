"""
Inference page for BuilderBrain Dashboard.

Interactive grammar-constrained generation and plan validation testing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import json
import time


def render_inference_page(data_loader, api_client):
    """Render the inference testing page."""

    st.header("🧠 Interactive Inference")

    # Model Selection
    col1, col2, col3 = st.columns(3)

    with col1:
        model_scales = api_client.get_model_scales()
        selected_scale = st.selectbox(
            "Model Scale",
            model_scales,
            index=model_scales.index("small") if "small" in model_scales else 0
        )

    with col2:
        grammar_constraints = api_client.get_grammar_constraints()
        available_grammars = grammar_constraints.get('available_grammars', [])

        selected_grammar = st.selectbox(
            "Grammar Type",
            available_grammars,
            index=0 if available_grammars else None
        )

    with col3:
        strict_mode = st.selectbox(
            "Constraint Mode",
            ["Strict", "Flexible"],
            index=0
        )

    # Inference Input
    st.subheader("📝 Input Prompt")

    input_prompt = st.text_area(
        "Enter your prompt:",
        value="Generate a JSON API call for user authentication",
        height=100
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        max_tokens = st.slider("Max Tokens", 50, 500, 150)

    with col2:
        if st.button("🚀 Generate Response", use_container_width=True, type="primary"):
            with st.spinner("Generating response..."):
                response = api_client.run_inference(
                    prompt=input_prompt,
                    model_scale=selected_scale,
                    grammar_strict=(strict_mode == "Strict"),
                    max_tokens=max_tokens
                )

            if "error" not in response:
                st.session_state.inference_result = response
                st.success("Response generated successfully!")
            else:
                st.error(f"Generation failed: {response['error']}")

    # Display Results
    if 'inference_result' in st.session_state:
        result = st.session_state.inference_result

        st.subheader("📋 Generation Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tokens Generated", result.get('tokens_generated', 0))

        with col2:
            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")

        with col3:
            violations = result.get('grammar_violations', 0)
            st.metric("Grammar Violations", violations)

        # Response Display
        st.subheader("Generated Response")
        response_text = result.get('response', '')

        # Color code based on grammar violations
        if violations == 0:
            st.success("✅ Grammar compliant")
        elif violations < 5:
            st.warning(f"⚠️ {violations} minor violations")
        else:
            st.error(f"❌ {violations} major violations")

        st.code(response_text, language='text')

        # Grammar Preview
        if selected_grammar:
            st.subheader("🔍 Grammar Preview")

            preview = api_client.get_grammar_preview(response_text, selected_grammar)

            if "error" not in preview:
                col1, col2 = st.columns(2)

                with col1:
                    st.text_area(
                        "Original Text",
                        preview.get('original_text', ''),
                        height=200,
                        disabled=True
                    )

                with col2:
                    constrained_text = preview.get('constrained_text', '')
                    if constrained_text != response_text:
                        st.text_area(
                            "Constrained Text",
                            constrained_text,
                            height=200,
                            disabled=True
                        )
                    else:
                        st.info("No constraints applied to this text")

                # Violations and Suggestions
                violations = preview.get('violations', [])
                if violations:
                    st.warning("Grammar Violations:")
                    for violation in violations:
                        st.error(f"• {violation}")

                suggestions = preview.get('suggestions', [])
                if suggestions:
                    st.info("Suggestions:")
                    for suggestion in suggestions:
                        st.info(f"• {suggestion}")
            else:
                st.error(f"Grammar preview failed: {preview['error']}")

    # Plan Validation Section
    st.subheader("🔧 Plan Validation")

    st.info("Test plan DAG validation and execution preview")

    # Sample plan input
    sample_plan = {
        "nodes": [
            {"id": "grasp", "type": "grasp", "params": {"object_id": "red_cube", "pose": {"x": 0.5, "y": 0.2, "z": 0.1}}},
            {"id": "rotate", "type": "rotate", "params": {"angle": 90, "axis": "z"}},
            {"id": "place", "type": "place", "params": {"target_pose": {"x": 0.3, "y": 0.4, "z": 0.2}}}
        ],
        "edges": [
            {"from": "grasp", "to": "rotate", "type": "seq"},
            {"from": "rotate", "to": "place", "type": "seq"}
        ]
    }

    # Plan input area
    plan_json = st.text_area(
        "Plan DAG (JSON)",
        value=json.dumps(sample_plan, indent=2),
        height=300
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Validate Plan", use_container_width=True):
            try:
                plan_dag = json.loads(plan_json)
                validation = api_client.validate_plan(plan_dag)

                if "error" not in validation:
                    st.session_state.plan_validation = validation
                    if validation.get('valid', False):
                        st.success("Plan validation successful!")
                    else:
                        st.error("Plan validation failed!")
                else:
                    st.error(f"Validation error: {validation['error']}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")

    with col2:
        if st.button("🔮 Preview Execution", use_container_width=True):
            try:
                plan_dag = json.loads(plan_json)
                preview = api_client.get_plan_execution_preview(plan_dag)

                if "error" not in preview:
                    st.session_state.execution_preview = preview
                    st.info("Execution preview generated!")
                else:
                    st.error(f"Preview error: {preview['error']}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")

    # Display Plan Results
    if 'plan_validation' in st.session_state:
        validation = st.session_state.plan_validation

        st.subheader("📋 Validation Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Valid", "✅ Yes" if validation.get('valid') else "❌ No")

        with col2:
            st.metric("Validation Time", f"{validation.get('validation_time', 0):.3f}s")

        with col3:
            errors = len(validation.get('errors', []))
            st.metric("Errors", errors)

        # Display errors and warnings
        errors = validation.get('errors', [])
        if errors:
            st.error("Validation Errors:")
            for error in errors:
                st.error(f"• {error}")

        warnings = validation.get('warnings', [])
        if warnings:
            st.warning("Validation Warnings:")
            for warning in warnings:
                st.warning(f"• {warning}")

    if 'execution_preview' in st.session_state:
        preview = st.session_state.execution_preview

        st.subheader("🔮 Execution Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Est. Execution Time", f"{preview.get('estimated_execution_time', 0):.2f}s")

            requirements = preview.get('resource_requirements', {})
            st.metric("CPU Required", f"{requirements.get('cpu', 0):.1%}")
            st.metric("Memory Required", f"{requirements.get('memory', 0):.1%}")

        with col2:
            risk = preview.get('risk_assessment', 'unknown')
            if risk == 'low':
                st.success(f"Risk Assessment: {risk.title()}")
            elif risk == 'medium':
                st.warning(f"Risk Assessment: {risk.title()}")
            else:
                st.error(f"Risk Assessment: {risk.title()}")

            suggestions = preview.get('optimization_suggestions', [])
            if suggestions:
                st.info("Optimization Suggestions:")
                for suggestion in suggestions:
                    st.info(f"• {suggestion}")

    # Grammar Testing Section
    st.subheader("🔤 Grammar Testing")

    col1, col2 = st.columns(2)

    with col1:
        test_text = st.text_area(
            "Test Text",
            value='{"name": "test", "value": 123}',
            height=150
        )

        if st.button("Test Grammar Compliance", use_container_width=True):
            preview = api_client.get_grammar_preview(test_text, selected_grammar)
            st.session_state.grammar_test = preview

    with col2:
        if 'grammar_test' in st.session_state:
            test_result = st.session_state.grammar_test

            if "error" not in test_result:
                violations = len(test_result.get('violations', []))
                if violations == 0:
                    st.success("✅ Grammar compliant")
                else:
                    st.error(f"❌ {violations} violations found")

                st.json(test_result)
            else:
                st.error(f"Grammar test failed: {test_result['error']}")

    # Export Options
    st.subheader("💾 Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export Response", use_container_width=True):
            if 'inference_result' in st.session_state:
                response_text = st.session_state.inference_result.get('response', '')
                st.download_button(
                    "Download Response",
                    response_text,
                    file_name="inference_response.txt",
                    use_container_width=True
                )

    with col2:
        if st.button("📋 Export Plan", use_container_width=True):
            st.download_button(
                "Download Plan",
                plan_json,
                file_name="test_plan.json",
                use_container_width=True
            )

    with col3:
        if st.button("🔧 Export Grammar Test", use_container_width=True):
            if 'grammar_test' in st.session_state:
                st.download_button(
                    "Download Grammar Test",
                    json.dumps(st.session_state.grammar_test, indent=2),
                    file_name="grammar_test.json",
                    use_container_width=True
                )
