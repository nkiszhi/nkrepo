# -*- coding: utf-8 -*-
"""
NKREPO Database Module

Provides database operations for malware sample management:
- Sample storage and retrieval
- Hash-based lookups
- Database sharding across 256 tables
"""

from .db_operations import DatabaseOperation

__all__ = [
    'DatabaseOperation',
]
