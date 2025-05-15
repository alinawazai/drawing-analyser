from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
from typing import Literal, Any


# ── ENUMS ──────────────────────────────────────────────────────────────
class DrawingType(BaseModel):
    drawing_type: Literal[
        "floor_plan", "section", "elevation", "structural", "mep", 
        "foundation", "fire_safety", "detail", "other"
    ]


# ── PURPOSE OF BUILDING USING LITERAL ─────────────────────────────────
class BuildingPurpose(BaseModel):
    purpose: Literal[
        "residential", "commercial", "mixed_use", "industrial", "infrastructure", "other"
    ]


# ── REUSABLE SMALL BLOCKS ─────────────────────────────────────────────
class StairsDetail(BaseModel):
    location: Optional[str] = ""
    purpose:  Optional[str] = ""


class ElevatorDetail(BaseModel):
    location: Optional[str] = ""
    purpose:  Optional[str] = ""


class HallwayDetail(BaseModel):
    location:    Optional[str] = ""
    approx_area: Optional[str] = ""



# ── FLATTENED SCHEMA ───────────────────────────────────────────────────
class DrawingMetadata(BaseModel):
    # ===== BASIC HEADERS =================================================
    drawing_type: DrawingType
    building_purpose: BuildingPurpose
    client_name: Optional[str] = ""
    project_title: Optional[str] = ""
    drawing_title: Optional[str] = ""

    # ===== CORE DETAILS ==================================================
    drawing_number: Optional[str] = ""
    project_number: Optional[str] = ""
    revision_number: Optional[int] = None
    scale: Optional[str] = ""
    architects: List[str] = Field(default_factory=list)

    notes_on_drawing: Optional[str] = ""
    table_on_drawing: Optional[str] = ""  # markdown table or ""

    # ===== FLOOR-PLAN & SPATIAL INFO ====================================
    communal_spaces: List[str] = Field(default_factory=list)  # e.g., ["hallways", "lounges"]
    private_spaces: List[str] = Field(default_factory=list)   # e.g., ["bedrooms", "bathrooms"]
    service_spaces: List[str] = Field(default_factory=list)   # e.g., ["kitchens", "utility rooms"]

    num_stairs: Optional[int] = None
    num_elevators: Optional[int] = None
    num_hallways: Optional[int] = None

    stairs_details: List[StairsDetail] = Field(default_factory=list)
    elevator_details: List[ElevatorDetail] = Field(default_factory=list)
    hallway_details: List[HallwayDetail] = Field(default_factory=list)

    # ===== SECTION-SPECIFIC FIELDS ======================================
    floor_heights: List[str] = Field(default_factory=list)  # e.g., ["GF-1: 3.3 m", …]
    structural_notes: Optional[str] = ""

    # ===== ELEVATION-SPECIFIC FIELDS ====================================
    facade_materials: List[str] = Field(default_factory=list)
    height_dims: List[str] = Field(default_factory=list)  # e.g., ["RL +45.800", "10 000 mm"]
    window_types: List[str] = Field(default_factory=list)  # e.g., ["double-glazed", "triple-glazed"]
    door_types: List[str] = Field(default_factory=list)  # e.g., ["fire-rated", "double-leaf"]
    roof_types: List[str] = Field(default_factory=list)  # e.g., ["flat", "pitched"]
    roof_materials: List[str] = Field(default_factory=list)  # e.g., ["tiles", "membrane"]
    balcony_types: List[str] = Field(default_factory=list)  # e.g., ["cantilevered", "loggia"]

    # ===== STRUCTURAL DRAWING FIELDS ====================================
    beam_schedule: Optional[str] = ""
    column_schedule: Optional[str] = ""
    rebar_notes: Optional[str] = ""
    foundation_notes: Optional[str] = ""
    structural_notes: Optional[str] = ""

    # ===== MEP DRAWING FIELDS ===========================================
    hvac_notes: Optional[str] = ""
    electrical_notes: Optional[str] = ""
    piping_notes: Optional[str] = ""
    plumbing_notes: Optional[str] = ""
    

    # ===== FOUNDATION FIELDS ============================================
    footing_type: Optional[str] = ""
    pile_count: Optional[int] = None
    depth_info: Optional[str] = ""
    soil_type: Optional[str] = ""
    drainage_info: Optional[str] = ""
    waterproofing_info: Optional[str] = ""
    
    # ===== FIRE SAFETY FIELDS ===========================================
    fire_exit_count: Optional[int] = None
    fire_exit_locations: List[str] = Field(default_factory=list)
    fire_alarm_system: Optional[str] = ""
    fire_suppression_system: Optional[str] = ""
    fire_rating: Optional[str] = ""
    
    # ===== DETAIL DRAWING FIELDS ========================================
    detail_type: Optional[str] = ""
    detail_number: Optional[str] = ""
    detail_description: Optional[str] = ""
    detail_notes: Optional[str] = ""
    detail_scale: Optional[str] = ""
    detail_dimensions: List[str] = Field(default_factory=list)  # e.g., ["200x200 mm", "3 m"]
    
    # ===== OTHER DRAWING FIELDS =========================================
    other_notes: Optional[str] = ""
    other_details: Dict[str, Any] = Field(default_factory=dict)
    
    # ===== ADDITIONAL FIELDS ============================================
    created_by: Optional[str] = ""
    created_on: Optional[str] = ""
    last_modified_by: Optional[str] = ""
    last_modified_on: Optional[str] = ""
    
    
