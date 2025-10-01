# OBSERVABILITY.md — builderbrain

## 0. purpose

Observability defines the metrics schema, dashboards, and alerting systems for BuilderBrain. This ensures comprehensive monitoring of system health, performance, and constraint satisfaction across all operational domains.

---

## 1. metrics schema

### 1.1 core metrics hierarchy

**system health:**
* **Availability:** uptime percentage, service health checks
* **Performance:** response times, throughput, error rates
* **Resource utilization:** CPU, GPU, memory, disk, network

**model performance:**
* **Accuracy:** task completion rates, constraint satisfaction
* **Efficiency:** tokens per second, energy per inference
* **Quality:** grammar compliance, plan execution success

**constraint monitoring:**
* **Grammar compliance:** CFG energy, token masking violations
* **Plan execution:** DAG validation rates, precondition failures
* **Safety invariants:** risk energy levels, constraint violations

### 1.2 metric definitions

**response time metrics:**
```python
class ResponseTimeMetrics:
    def __init__(self):
        self.p50 = 0.0    # Median response time
        self.p95 = 0.0    # 95th percentile
        self.p99 = 0.0    # 99th percentile
        self.mean = 0.0   # Average response time

    def update(self, response_time: float):
        # Update percentile calculations
        # Rolling window or exponential moving average
        pass
```

**constraint satisfaction metrics:**
```python
class ConstraintMetrics:
    def __init__(self):
        self.grammar_compliance_rate = 0.0
        self.plan_execution_success_rate = 0.0
        self.constraint_violation_rate = 0.0
        self.risk_energy_average = 0.0
        self.dual_variable_values = {}
```

---

## 2. logging architecture

### 2.1 log levels and categories

**log levels:**
* **DEBUG:** detailed internal state, debugging information
* **INFO:** normal operational messages, milestones
* **WARN:** recoverable issues, performance concerns
* **ERROR:** failures requiring attention, constraint violations
* **CRITICAL:** safety violations, system failures

**log categories:**
* **Training:** model updates, loss values, constraint evolution
* **Inference:** request processing, response generation
* **Safety:** risk energy monitoring, constraint violations
* **Performance:** resource utilization, timing metrics
* **Audit:** access logs, configuration changes

### 2.2 structured logging

**log format:**
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "category": "inference",
  "component": "dual_rail",
  "message": "Grammar constraint satisfied",
  "metrics": {
    "grammar_energy": 0.15,
    "constraint_satisfaction": 0.95,
    "response_time": 1.23
  },
  "context": {
    "request_id": "req_12345",
    "user_id": "user_67890",
    "domain": "api_json",
    "model_version": "v2.1.0"
  },
  "trace_id": "trace_abcdef123456"
}
```

**implementation:**
```python
import json
import logging

class StructuredLogger:
    def __init__(self, service_name: str):
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)

    def log_metric(self, level: str, message: str, **kwargs):
        log_entry = {
            "timestamp": self._get_timestamp(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logger.log(self._level_to_int(level), json.dumps(log_entry))

    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
```

---

## 3. dashboards and visualization

### 3.1 real-time dashboards

**system health dashboard:**
* Service availability indicators
* Error rate trends (last 24 hours)
* Resource utilization graphs
* Active alert status

**performance dashboard:**
* Response time distributions (P50/P95/P99)
* Throughput metrics (requests/second)
* Queue lengths and processing delays
* Model inference latency breakdown

**constraint monitoring dashboard:**
* Grammar compliance rates over time
* Plan execution success rates
* Dual variable evolution graphs
* Constraint violation alerts

### 3.2 alerting dashboards

**alert severity levels:**
* **P0 (Critical):** immediate human response required
* **P1 (High):** response within 1 hour
* **P2 (Medium):** response within 4 hours
* **P3 (Low):** response within 24 hours

**alert types:**
* **Safety alerts:** risk energy spikes, constraint violations
* **Performance alerts:** latency degradation, error rate increases
* **Resource alerts:** memory/CPU utilization above thresholds
* **Quality alerts:** grammar compliance drops, plan failures

### 3.3 visualization components

**time series charts:**
* Moving averages and trend lines
* Anomaly detection overlays
* Confidence intervals and thresholds

**distribution plots:**
* Response time histograms
* Error rate distributions
* Constraint satisfaction histograms

**correlation matrices:**
* Metric correlations for root cause analysis
* Performance vs constraint satisfaction
* Resource utilization vs error rates

---

## 4. alerting system

### 4.1 alert configuration

**alert rules:**
```yaml
alerts:
  - name: high_error_rate
    condition: "error_rate > 0.05"
    duration: "5m"
    severity: "P1"
    channels: ["slack", "email", "pager"]

  - name: constraint_violation_spike
    condition: "constraint_violations > baseline + 3*stddev"
    duration: "10m"
    severity: "P0"
    channels: ["slack", "email", "pager", "sms"]

  - name: safety_risk_increase
    condition: "risk_energy_p95 > threshold"
    duration: "1m"
    severity: "P0"
    channels: ["slack", "email", "pager", "sms"]
```

### 4.2 alert routing

**notification channels:**
* **Slack:** real-time team notifications
* **Email:** detailed incident reports
* **Pager/SMS:** critical alerts for on-call engineers
* **Webhook:** integration with external monitoring systems

**escalation policies:**
* P0 alerts: immediate on-call engineer notification
* P1 alerts: team lead notification within 15 minutes
* P2 alerts: daily standup discussion
* P3 alerts: weekly review meeting

### 4.3 alert lifecycle

**alert states:**
1. **Triggered:** condition met, notification sent
2. **Acknowledged:** engineer confirms awareness
3. **Investigating:** active troubleshooting in progress
4. **Resolved:** issue fixed, metrics back to normal
5. **Closed:** incident documented and lessons learned

**auto-resolution:**
* Self-healing mechanisms for known issues
* Automatic fallback activation
* Graceful degradation detection

---

## 5. telemetry collection

### 5.1 metric collection points

**application metrics:**
* Request/response cycle timing
* Model inference duration
* Constraint evaluation time
* Error occurrence and recovery

**infrastructure metrics:**
* CPU, GPU, memory utilization
* Network I/O and bandwidth
* Disk space and I/O operations
* Container and service health

**business metrics:**
* User satisfaction scores
* Task completion rates
* Constraint satisfaction percentages
* Cost per inference metrics

### 5.2 sampling strategies

**sampling rates:**
* High-frequency metrics (response times): 100% sampling
* Medium-frequency metrics (error rates): 10% sampling
* Low-frequency metrics (resource utilization): 1% sampling

**adaptive sampling:**
* Increase sampling during incidents
* Decrease sampling during normal operation
* Focus sampling on critical paths

---

## 6. distributed tracing

### 6.1 trace structure

**trace components:**
* **Trace ID:** unique identifier for request lifecycle
* **Span ID:** individual operation within trace
* **Parent Span ID:** hierarchical relationship tracking

**trace propagation:**
* HTTP headers for cross-service traces
* Message queue headers for async operations
* Database query annotations for data operations

### 6.2 trace visualization

**flame graphs:**
* Execution time breakdown by component
* Bottleneck identification
* Optimization opportunity detection

**dependency graphs:**
* Service interaction visualization
* Latency attribution across systems
* Failure propagation analysis

---

## 7. performance monitoring

### 7.1 latency tracking

**end-to-end latency:**
* Request initiation to response completion
* Component-level timing breakdown
* Network and I/O contribution analysis

**percentile analysis:**
* P50: typical user experience
* P95: performance under moderate load
* P99: worst-case performance
* P99.9: extreme outlier handling

### 7.2 throughput monitoring

**request throughput:**
* Requests per second by endpoint
* Concurrent request handling capacity
* Queue depth and processing rates

**resource throughput:**
* Tokens processed per second
* Model inferences per GPU
* Data transfer rates
* Storage I/O operations

### 7.3 error rate monitoring

**error classification:**
* **4xx errors:** client-side issues (malformed requests)
* **5xx errors:** server-side failures (system issues)
* **Constraint errors:** grammar/plan violations
* **Safety errors:** risk threshold violations

**error correlation:**
* Error patterns by time of day
* Error correlation with load levels
* Error propagation across components

---

## 8. constraint observability

### 8.1 grammar compliance monitoring

**real-time metrics:**
* Current grammar energy levels
* Token masking violation rates
* Parse failure frequencies

**historical trends:**
* Grammar compliance over time
* Impact of model updates on compliance
* Domain-specific compliance patterns

### 8.2 plan execution monitoring

**execution metrics:**
* Plan validation success rates
* Node execution success rates
* Precondition satisfaction rates
* Resource constraint violations

**plan quality metrics:**
* Plan complexity distribution
* Execution time variance
* Error recovery effectiveness

### 8.3 dual variable monitoring

**constraint evolution:**
* Dual variable trajectories over time
* Constraint target vs actual performance
* Multi-objective trade-off analysis

**optimization health:**
* Gradient conflict detection
* Constraint satisfaction stability
* Learning rate adaptation effectiveness

---

## 9. operational dashboards

### 9.1 executive dashboard

**high-level KPIs:**
* Overall system availability
* User satisfaction trends
* Constraint compliance rates
* Performance vs SLA targets

**business impact:**
* Cost per request trends
* Resource utilization efficiency
* Error resolution times
* User engagement metrics

### 9.2 engineering dashboard

**technical metrics:**
* Model performance by domain
* Constraint satisfaction by type
* Resource utilization by component
* Error patterns and frequencies

**debugging aids:**
* Recent error samples
* Performance regression alerts
* Constraint violation examples
* System health indicators

### 9.3 operations dashboard

**infrastructure monitoring:**
* Server health and resource usage
* Network connectivity status
* Database performance metrics
* External service dependencies

**deployment tracking:**
* Version rollout progress
* Feature flag status
* Configuration change impact
* Rollback readiness

---

## 10. anomaly detection

### 10.1 statistical anomaly detection

**baseline establishment:**
* Rolling window statistical baselines
* Seasonal pattern recognition
* Domain-specific normal ranges

**anomaly scoring:**
* Z-score based outlier detection
* Mahalanobis distance for multivariate anomalies
* Time series anomaly detection algorithms

### 10.2 machine learning anomaly detection

**unsupervised learning:**
* Autoencoder-based anomaly detection
* Isolation forest algorithms
* One-class SVM approaches

**supervised learning:**
* Historical anomaly labeling
* Pattern recognition for known issues
* Predictive failure detection

### 10.3 alert thresholds

**static thresholds:**
* Hard-coded limits for critical metrics
* Configurable warning levels
* Environment-specific adjustments

**dynamic thresholds:**
* Adaptive baselines based on historical data
* Seasonal and trend-aware limits
* Context-aware threshold adjustment

---

## 11. incident response

### 11.1 incident classification

**severity assessment:**
* **P0:** system unavailable, safety violations
* **P1:** significant performance degradation
* **P2:** minor functionality issues
* **P3:** cosmetic or minor issues

**impact assessment:**
* User-facing impact scope
* Business process disruption
* Data integrity concerns
* Regulatory compliance risks

### 11.2 response procedures

**automated responses:**
* Circuit breaker activation
* Automatic fallback mechanisms
* Load balancer rerouting
* Cache warming for recovery

**manual interventions:**
* Configuration adjustments
* Model rollback procedures
* Infrastructure scaling
* External service coordination

### 11.3 post-incident analysis

**root cause analysis:**
* Timeline reconstruction
* Contributing factor identification
* Failure mode classification
* Mitigation strategy development

**lessons learned:**
* Process improvement recommendations
* Monitoring enhancement suggestions
* Training and documentation updates
* Preventive measure implementation

---

## 12. data retention and compliance

### 12.1 log retention policies

**retention by log type:**
* **Application logs:** 90 days
* **Security logs:** 2 years
* **Audit logs:** 7 years
* **Performance logs:** 1 year

**retention enforcement:**
* Automated log rotation and archiving
* Compliance with data protection regulations
* Secure deletion procedures
* Legal hold capabilities

### 12.2 data export and reporting

**compliance reporting:**
* GDPR data subject access requests
* SOX audit trail exports
* Regulatory compliance reports
* Internal audit requirements

**data portability:**
* Structured data export formats
* API-based data access
* Bulk download capabilities
* Third-party integration support

---

## 13. future observability enhancements

**planned improvements:**
* Advanced anomaly detection with deep learning
* Predictive monitoring and alerting
* Automated root cause analysis
* Self-healing observability systems

**research directions:**
* Causality inference from observability data
* Multi-modal observability (text, metrics, traces)
* Federated observability across distributed systems
* AI-powered observability assistants
