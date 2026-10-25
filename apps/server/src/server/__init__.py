"""codenames-server: the authoritative real-time game server.

Holds full game state, validates every clue/guess via codenames_core, and sends each
client only the view for its role. Installs codemaster without the `pipeline` extra.
"""
