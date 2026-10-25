"""ONLINE: loads an exported sense artifact and serves similarity queries.

numpy + codenames_core only. This is the code the live website runs at game time, so it
must never import torch, transformers, or anything from `codemaster.pipeline`.
"""
