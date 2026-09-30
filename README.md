# TPUv4-Style Resource Allocation and Resiliency Simulator

This project contains a small Python Monte Carlo simulator inspired by the paper:

> “Resiliency at Scale: Managing Google’s TPUv4 Machine Learning Supercomputer”

The simulator demonstrates why a large distributed machine-learning job may fail to start even when a cloud system has many accelerators available.

## What the Simulator Models

The simulator represents an infrastructure containing 64 TPU cubes.

Each cube can be:

- Free and available for a new job
- Failed because of a hardware or infrastructure failure
- Occupied by another workload

The simulator compares two allocation strategies.

### Static Allocation

Resources are divided into fixed groups. A distributed job succeeds only when one fixed group contains enough healthy and free cubes.

This represents a system where resource placement and connectivity are mostly predetermined.

### Reconfigurable Allocation

A distributed job can use any healthy and free cubes in the resource pool.

This represents TPUv4-style cube-level reconfiguration, where healthy resources can be logically connected even if they are not physically adjacent.

## Simulator Metric

The simulator measures the percentage of successful job requests:

```text
Success Rate =
(Number of successful requests / Total simulation trials) × 100
