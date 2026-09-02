# Optimizing Delivery ETAs with Graph-Based Network Intelligence

## Overview

Delivery ETA prediction is often treated as a route-level regression problem, where factors such as distance and time of day are used to estimate travel time.

This project explores a different approach: **modeling the delivery network as a graph** and incorporating the structure and operational behavior of that network into ETA prediction.

Distribution centers are represented as **nodes**, while movement between centers is represented as **directed edges**. Graph-derived information is then combined with temporal and operational features and passed to a machine learning model for continuous ETA prediction.

The central question explored in this project is:

> **Can understanding the structure and historical behavior of the delivery network improve ETA predictions beyond a conventional distance-based model?**

---

## Core Approach

The project builds a hybrid **graph-temporal ETA prediction system** consisting of four major components:

### 1. Delivery Network as a Directed Graph

The logistics network is represented as a directed graph:

* **Nodes** → distribution / logistics centers
* **Edges** → observed routes between centers
* **Direction** → source center → destination center

This representation makes it possible to capture information that is not explicitly present in individual delivery records, such as network connectivity and hub importance.

---

### 2. Graph Representation Learning

#### Node2Vec

`Node2Vec` is used to generate low-dimensional embeddings for each network node.

The embeddings capture structural relationships in the delivery network through random walks, allowing geographically or operationally related centers to be represented in a learned feature space.

The model uses:

* 16-dimensional node embeddings
* Random-walk based structural exploration
* Window-based embedding learning

These embeddings are incorporated separately for the source and destination centers of each delivery leg.

#### Network Centrality

Additional structural information is extracted using:

* **Betweenness centrality**
* **In-degree**
* **Out-degree**

Betweenness centrality provides an indication of whether a center acts as an important intermediary within the network, while degree-based features describe its connectivity.

---

### 3. Operational & Temporal Intelligence

Graph structure alone does not describe how the network behaves operationally.

Historical delivery performance is therefore incorporated into the feature representation.

For each source and destination center, historical **delay ratios** are calculated and used as operational indicators of how reliably a node tends to perform.

A similar historical delay factor is also calculated for routes.

#### GraphSAGE-style Neighborhood Aggregation

The project further incorporates local neighborhood information through a **GraphSAGE-style message-passing approach**.

For each node, the historical delay characteristics of its neighboring nodes are aggregated using neighborhood averaging.

This allows the model to capture the idea that:

> **The operational behavior of a logistics center can be influenced not only by the center itself, but also by the surrounding network.**

---

### 4. Temporal Encoding

Time of day is represented using cyclical encoding:

```text
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

This preserves the cyclic relationship between hours — for example, the transition from 23:00 to 00:00.

The final feature representation combines:

```text
Route Distance
      +
Temporal Features
      +
Node2Vec Embeddings
      +
Network Centrality
      +
Historical Node Delays
      +
Neighborhood Information
      +
Graph × Time Interactions
```

---

## Graph-Temporal Feature Construction

A key part of the project is the construction of a **master hybrid feature matrix**.

For each delivery leg, the model receives graph information associated with both the source and destination nodes.

The representation includes:

* Source Node2Vec embedding
* Destination Node2Vec embedding
* Source degree / betweenness measures
* Destination degree / betweenness measures
* Historical source delay
* Historical destination delay
* GraphSAGE-style neighborhood features
* Time-of-day features
* Graph-temporal cross-products

The graph-temporal interaction terms explicitly combine historical delay behavior with time:

```text
Source Delay × sin(Time)
Source Delay × cos(Time)

Destination Delay × sin(Time)
Destination Delay × cos(Time)
```

This allows the model to represent situations where the effect of a particular network location changes depending on the dispatch time.

---

## Baseline vs Graph-Enhanced Model

To measure the contribution of graph intelligence, two models are compared.

### Baseline

The baseline model uses a minimal set of conventional features:

```text
Route Distance
Hour of Day
```

### Graph-Enhanced Model

The enhanced model combines the baseline features with:

```text
Node2Vec embeddings
Network centrality
Historical delay factors
GraphSAGE-style neighborhood aggregation
Graph-temporal interactions
Route type
```

Both models use the same train-test split so that the effect of the additional graph information can be evaluated directly.

---

## Machine Learning Model

The final ETA prediction model uses a **Gradient Boosting Regressor**.

Gradient boosting was chosen to model the complex non-linear relationships between:

* Route characteristics
* Network structure
* Historical delays
* Temporal effects
* Graph-derived representations

The notebook also experiments with different boosting configurations, including `HistGradientBoostingRegressor`, before arriving at the final graph-enhanced configuration.

---

## Evaluation

The project evaluates the system using both conventional regression metrics and operational ETA metrics.

### Continuous ETA Prediction

The final graph-enhanced model achieved:

| Metric             |        Result |
| ------------------ | ------------: |
| R²                 |     **94.5%** |
| MAE                | **~32.4 min** |
| ETA within ±5 min  |      **~20%** |
| ETA within ±15 min |      **~52%** |
| ETA within ±30 min |      **~76%** |

These metrics evaluate how close the predicted arrival time is to the actual observed arrival time.

### Improvement over Baseline

The graph-enhanced feature representation substantially improved the strict ETA-window accuracy compared with the baseline.

The final experiments reported approximately:

```text
Baseline within ±15% : ~29.8%
Graph-enhanced       : ~49.0%

Improvement          : ~19 percentage points
```

The model also reduced average prediction error by roughly **18 minutes per trip** in the final experiments.

---

## Operational Risk Analysis

Beyond predicting a continuous ETA, the project also evaluates whether a delivery is likely to exceed its planned travel-time threshold.

This creates an operational **risk-audit layer** on top of the ETA model.

The system categorizes predicted errors into:

```text
Within ±15%       → Precise ETA window
15–30% deviation  → Marginal buffer
>30% deviation    → Critical service failure
```

This provides a more operational interpretation of model performance than regression metrics alone.

---

## Network Bottleneck Analysis

The graph representation is also used independently of the ML model to identify potentially important areas of the logistics network.

Using network centrality and historical delay information, the project investigates:

* Highly connected hubs
* Network chokepoints
* High-delay corridors
* High-volume problematic routes
* Sources of SLA breaches

This shifts the project from simply predicting ETAs toward **understanding why and where delivery delays occur**.

---

## Live Inference Concept

The notebook also demonstrates a deployment-oriented inference interface where a user can provide:

```text
Source Center
Destination Center
Planned Route Distance
Dispatch Hour
Route Type
```

The system then generates an ETA/risk assessment and can compare the expected performance against the existing route-time estimate.

This demonstrates how the graph-enhanced model could act as a **decision-support layer for logistics operations** rather than only functioning as an offline prediction model.

---

## System Architecture

```text
                 DELIVERY DATA
                       │
                       ▼
              Route-Level Features
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Temporal Features          Delivery Network
                                     │
                                     ▼
                              Directed Graph
                                     │
                ┌────────────────────┼───────────────────┐
                │                    │                   │
                ▼                    ▼                   ▼
             Node2Vec           Centrality         Neighborhood
             Embeddings          Metrics            Aggregation
                │                    │                   │
                └────────────────────┼───────────────────┘
                                     │
                                     ▼
                           Historical Delay Factors
                                     │
                                     ▼
                          Graph × Temporal Features
                                     │
                                     ▼
                          Hybrid Feature Matrix
                                     │
                                     ▼
                         Gradient Boosting Model
                                     │
                                     ▼
                              ETA Prediction
                                     │
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
                Continuous ETA              Risk Assessment
                       │                           │
                       ▼                           ▼
                 Actual ETA                 SLA / Delay Risk
```

---

## Technologies

* Python
* NumPy
* Pandas
* NetworkX
* Node2Vec
* Scikit-learn
* Matplotlib
* Plotly

### Key Concepts

* Graph-based machine learning
* Node embeddings
* Node2Vec
* GraphSAGE-style message passing
* Network centrality
* Feature engineering
* Time-series / cyclical encoding
* Gradient boosting
* ETA regression
* Logistics network analysis

---

## Key Takeaway

The main idea of this project is that **delivery time is not determined only by the individual route**.

A route exists within a larger logistics network, and its performance can be influenced by:

* The structural position of its source and destination hubs
* Connectivity within the network
* Historical delays at nearby hubs
* Route-specific operational behavior
* Time of day
* Interactions between network conditions and temporal patterns

By incorporating these relationships into the feature space, the graph-enhanced model provides a substantially more informative representation of delivery conditions than a simple distance-based ETA model.

The project therefore combines **graph representation learning, network analysis, and supervised regression** to build a more context-aware ETA prediction and operational risk system.
