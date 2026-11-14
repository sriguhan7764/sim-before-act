"""ACT (Action Chunking Transformer) pilot pipeline: dataset conversion, training, offline inference, action segments.

OFFLINE ONLY. Nothing in this package opens a serial port, camera or any device, and nothing imports a LeRobot robot
connect path. ``torch`` / ``lerobot`` are imported lazily inside the functions that need them (``train``, ``infer``,
``adapter``) so that the conversion and action-segment code runs in the plain collector environment.

Modules
  contract   joint order / units / labels shared by every artefact
  align      time alignment rules (pure python)
  convert    dataset v0.1 manifest + split -> training-ready converted dataset (+ audit, lineage check)
  load       numpy reader of a converted dataset and the action-chunk (padding) builder
  segment    ``action-segment-v0.1`` writer / validator (pure python + numpy)
  synthetic  tiny SYNTHETIC fixture builder (never real demonstrations)
  adapter    torch Dataset feeding ``ACTPolicy.forward``
  train      training / validation / checkpoint selection / resume / freeze
  infer      offline inference from a frozen checkpoint -> action segment
CLI: ``python -m collector.act --help``.
"""
