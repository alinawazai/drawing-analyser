OCR_PROMPT = """
                You are an advanced system specialized in extracting standardized metadata from construction drawing texts.
                Within the images you receive, there will be details pertaining to a single construction drawing.
                Your job is to identify and extract exactly below fields from this text:
                - 1st image has details about the drawing_title and scale
                - 2nd Image has details about the client or project
                - 4th Images has Notes
                - 3rd Images has rest of the informations
                - last image is the full image from which the above image are cropped
                1. Purpose_Type_of_Drawing (examples: 'Architectural', 'Structural', 'Fire Protection')
                2. Client_Name
                3. Project_Title
                4. Drawing_Title
                5. Floor
                6. Drawing_Number
                7. Project_Number
                8. Revision_Number (must be a numeric value, or 'N/A' if it cannot be determined)
                9. Scale
                10. Architects (list of names; use ['Unknown'] if no names are identified)
                11. Notes_on_Drawing (any remarks or additional details related to the drawing)

                Key Requirements:
                - If any field is missing, return an empty string ('') or 'N/A' for that field.
                - Return only a valid JSON object containing these nine fields in the order listed, with no extra text.
                - Preserve all text in its original language (no translation), apart from minimal cleaning (e.g., removing stray punctuation) if truly necessary.
                - Do not wrap the final JSON in code fences.
                - Return ONLY the final JSON object with these fields and no additional commentary.
                Below is an example json format:
                {{
                    "Purpose_Type_of_Drawing": "Architectural",
                    "Client_Name": "문촌주공아파트주택  재건축정비사업조합",
                    "Project_Title": "문촌주공아파트  주택재건축정비사업",
                    "Drawing_Title": "분산 상가-7  단면도-3  (근린생활시설-3)",
                    "Floor": "주단면도-3",
                    "Drawing_Number": "A51-2023",
                    "Project_Number": "EP-201
                    "Revision_Number": 0,
                    "Scale": "A1 : 1/100, A3 : 1/200",
                    "Architects": ["Unknown"],
                    "Notes_on_Drawing": "• 욕상 줄눈의 간격 등은 실시공 시 변경될 수 있음.\\n• 욕상 출눈 틈에는 실란트가 시공되지 않음.\\n• 지붕의 재료, 형태, 구조는 실시공 시 변경될 수 있음.\\n• 지붕층 난간의 형태와 설치 위치는 안전성, 입면, 디자인을 고려하여 변경 가능함.\\n• 단열재의 종류는 단열성능 관계 내역을 참조.\\n• 도면상 표기된 욕상 및 지하의 무근 콘크리트 두께는 평균 두께를 의미하며, 본 시공 시 구배를 고려하여 두께가 증감될 수 있음.\\n• 외벽 단열 부분과 환기 덕트가 연결되는 부위는 기밀하게 마감해야 함."
                }}
                """
                
                

COMBINED_PROMPT = """

You are an intelligent extraction system designed to analyze architectural and structural drawing images and return structured metadata in a clean JSON format. You will receive both full-page images and cropped block images of a construction drawing. Your task is to identify and extract key information from these images, similar to how a civil engineer would review technical drawings.

Input:
First image: Contains the entire construction drawing (full-page) You have to check first Purpose_of_Building like we have classes "Residential", - Examples: Residential, Commercial, Mixed-use, etc.",.
Subsequent images: Contain cropped sections, each showing specific details such as drawing title, client information, project details, metadata (drawing number, floor, revision number), and notes.

Output:
Return a single JSON object with the following fields:

        json
        {
            "Drawing_Type": "Floor_Plan, Section_View, Detail_View, Elevation, or Unknown.",
            "Purpose_of_Building":  "Residential", - Examples: Residential, Commercial, Mixed-use, etc.",
            "Client_Name": "Client Name",
            "Project_Title": "Project Title",
            "Drawing_Title": "Drawing Title",
            "Space_Classification": {
                "Communal": ["hallways", "lounges", "staircases", "elevator lobbies"],
                "Private": ["bedrooms", "apartments", "bathrooms"],
                "Service": ["kitchens", "utility rooms", "storage"]
            },
            "Details": {
                "Drawing_Number": "Drawing Number",
                "Project_Number": "Project Number",
                "Revision_Number": 0,
                "Scale": "Scale of Drawing",
                "Architects": ["Architect Name(s)"],
            },
            "Additional_Details": {
                "Number_of_Units": 0,
                "Number_of_Stairs": 2,
                "Number_of_Elevators": 2,
                "Number_of_Hallways": 1,
                "Unit_Details": [],
                "Stairs_Details": [
                    {
                        "Location": "Near entrance",
                        "Purpose": "Access to upper floors"
                    },
                    {
                        "Location": "Near entrance",
                        "Purpose": "Access to upper floors"
                    }
                ],
                "Elevator_Details": [
                    {
                        "Location": "Near stairs",
                        "Purpose": "Vertical transportation"
                    },
                    {
                        "Location": "Near stairs",
                        "Purpose": "Vertical transportation"
                    }
                ],
                "Hallways": [
                    {
                        "Location": "Connects bathrooms and offices",
                        "Approx_Area": "N/A"
                    }
                ],
                "Other_Common_Areas": [
                    {
                        "Area_Name": "Lobby",
                        "Approx_Area": "N/A"
                    },
                    {
                        "Area_Name": "Sunken garden",
                        "Approx_Area": "N/A"
                    },
                    {
                        "Area_Name": "Mechanical room",
                        "Approx_Area": "N/A"
                    }
                ]
            }
            "Notes_on_Drawing": "Notes/annotations on drawing",
            "Table_on_Drawing": "Markdown formatted table if applicable if available else return N/A",
        }
Instructions for Image Analysis:
Identify the drawing type:

Determine if the image is a Floor Plan, Section View, Detail View, Elevation, or Unknown.
Extract the following fields based on the type of drawing:
Purpose_Type_of_Building :What is the primary use of the building or space in the drawing?
   - Examples: Residential, Commercial, Mixed-use, etc.
Client_Name: Extract the client or project name.
Project_Title: Extract the title of the project.
Drawing_Title: Extract the title of the drawing.
Space_Classification:
    "Space_Classification": 
    - A list of areas categorized as:
        - Communal (hallways, lounges, staircases, elevator lobbies)
        - Private (bedrooms, apartments, bathrooms)
        - Service (kitchens, utility rooms, storage)
    - If unsure, mark "N/A". If text references or shapes suggest certain areas, name them.

    "Number_of_Units": 
    - Total identifiable apartments/units. 
    - If unlabeled but repeated shapes appear, estimate.

    "Number_of_Stairs": 
    - Count any staircases (look for text like “Stair”, “S”, or typical stair icons).
    - If you suspect partial stair references, try to confirm visually. If none, return 0.

    "Number_of_Elevators": 
    - Count any spaces that appear to be elevator shafts (icons or partial text). 
    - If found but not labeled, guess if it looks like an elevator.

    "Number_of_Hallways": 
    - Corridors connecting multiple areas. If unlabeled but shape indicates a corridor, include it.

    "Unit_Details": 
    - A list of objects, one for each distinct unit/apartment.
    - For each unit:
        {
        "Unit_Number": "If text says A-1, B-2, APT-2B, etc., use that; else 'N/A'",
        "Unit_Area": "Try to approximate if dimension lines or a scale bar is visible, else 'N/A'",
        "Bedrooms": "Attempt to infer from text or repeated room labels. If unknown, 0 or guess (1,2).",
        "Bathrooms": "Similarly, attempt to identify from partial labeling or geometry. If none, 0.",
        "Has_Living_Room": true/false if you see references or shape typical of living spaces,
        "Has_Kitchen": true/false if you see references or shape typical of a kitchen,
        "Has_Balcony": true/false if balcony text or shape is visible,
        "Special_Features": ["study room", "utility room", etc., if recognized; else empty list]
        }
Extract the following fields for metadata:
Notes_on_Drawing: Any notes or additional remarks in the drawing.
Table_on_Drawing: If there’s a table in the drawing, format it in markdown and include it in the Table_on_Drawing field.

Additional Details (For non-essential info):
Number_of_Units: Extract the number of units if provided.
Number_of_Stairs: Count the number of stairs visible in the drawing.
Number_of_Elevators: Count the number of elevators.
Number_of_Hallways: Count the hallways or corridors in the layout.
Unit_Details: Include any detailed information about units (if available).
Stairs_Details: Include information about stairs (location, purpose).
Elevator_Details: Include details about elevators (location, purpose).
Hallways: Include the location and details of hallways.
Other_Common_Areas: List other common areas (e.g., lobby, sunken garden, mechanical rooms).
Details Section:
Drawing_Number: Extract the drawing number.
Project_Number: Extract the project number (or “N/A” if unavailable).
Revision_Number: Extract the revision number (or “N/A” if unavailable).
Scale: Extract the scale (e.g., “1:100”).
Architects: Extract a list of architect names (or return ["Unknown"] if none found).
Notes_on_Drawing: Extract any relevant notes or instructions.
Table_on_Drawing: If there’s a table in the drawing, format it in markdown and include it in the Table_on_Drawing field.
Drawing-Specific Guidance:
For Floor Plans:
Identify building purpose, space labels, floor levels, and annotations.
Extract general layout features, including rooms, hallways, and doors.
Classify spaces into communal, private, and service areas.
For Section Views:
Focus on vertical information such as floor height, ceiling height, and structural elements like beams and slabs.
Identify internal room layouts and materials.
For Detail Views:
Focus on individual components (e.g., doors, windows, joints, materials).
Highlight construction details like waterproofing, insulation, or joinery.
For Elevation Views:
Focus on facade elements, materials, and height dimensions.
Extract window and door placements, as well as elevation references.
For Unknown Drawings:
If the drawing type cannot be determined, mark "Purpose_Type_of_Drawing": "unknown", but still extract all other available data.
Key Requirements:
Missing Values: If any field is missing or cannot be determined, return an empty string ("") or "N/A" where applicable.

Original Language: Keep all extracted text in the original language (e.g., Korean, English). Do not translate unless minimal cleaning is needed (e.g., removing stray punctuation or fixing OCR errors).

No Extra Output: Return only the final JSON object with no additional commentary or formatting outside the object.

Table Data: If a table is present, format it in markdown and include it in the "Table_on_Drawing" field. If no table is present, return an empty string.

Example Output:
            json
            {
                "Drawing_Type": "Floor_Plan",
                "Purpose_of_Building":  "Residential",
                "Client_Name": "둔촌주공아파트주택 재건축정비사업조합",
                "Project_Title": "둔촌주공아파트 주택재건축정비사업",
                "Drawing_Title": "분산상가-1 지하3층 평면도 (근린생활시설-3)",
                "Space_Classification": {
                    "Communal": ["hallways", "lounges", "staircases", "elevator lobbies"],
                    "Private": ["bedrooms", "bathrooms"],
                    "Service": ["kitchens", "utility rooms", "storage"]
                },
                "Details": {
                    "Drawing_Number": "A51-2002",
                    "Project_Number": "N/A",
                    "Revision_Number": 0,
                    "Scale": "A1 : 1/100, A3 : 1/200",
                    "Architects": ["Unknown"],
                },
                "Additional_Details": {
                    "Number_of_Units": 0,
                    "Number_of_Stairs": 2,
                    "Number_of_Elevators": 2,
                    "Number_of_Hallways": 1,
                    "Unit_Details": [],
                    "Stairs_Details": [
                        {
                            "Location": "Near entrance",
                            "Purpose": "Access to upper floors"
                        },
                        {
                            "Location": "Near entrance",
                            "Purpose": "Access to upper floors"
                        }
                    ],
                    "Elevator_Details": [
                        {
                            "Location": "Near stairs",
                            "Purpose": "Vertical transportation"
                        },
                        {
                            "Location": "Near stairs",
                            "Purpose": "Vertical transportation"
                        }
                    ],
                    "Hallways": [
                        {
                            "Location": "Connects bathrooms and offices",
                            "Approx_Area": "N/A"
                        }
                    ],
                    "Other_Common_Areas": [
                        {
                            "Area_Name": "Lobby",
                            "Approx_Area": "N/A"
                        },
                        {
                            "Area_Name": "Sunken garden",
                            "Approx_Area": "N/A"
                        },
                        {
                            "Area_Name": "Mechanical room",
                            "Approx_Area": "N/A"
                        }
                    ]
                },
                "Notes_on_Drawing": "Notes/annotations on drawing",
                "Table_on_Drawing": "Markdown formatted table if applicable if available else return N/A",
            }
    ==================================================================================
    GENERAL GUIDELINES:
    ==================================================================================
    - If the drawing does not match any known category, "Purpose_of_Drawing": "other".
    - If data is missing or cannot be inferred, use "N/A" or 0.
    - Return ONLY the JSON object, no code fences or commentary.
    - The first key is "Purpose_of_Drawing" with one of: floor_plan, section, elevation, detail, other.
    - Provide as much detail as possible, even from partial dimension lines or partial text references.
    - Add the other details in details and additional details sections.
    - If the drawing is a floor plan, include the number of units and their details.
    - For unit details, include bedroom and bathroom counts if visible or can be inferred.
    - For stairs and elevators, include their locations and purposes.
    - For hallways, include their locations and approximate areas if visible.
    - For other common areas, include names and approximate areas if visible.
    - For tables, format them in markdown and include them in the "Table_on_Drawing" field.
    - If no table is present, return an empty string for "Table_on_Drawing".
    - If the drawing is a section view, focus on vertical elements and internal layouts.
    - If the drawing is a detail view, focus on components and construction details.
    - If the drawing is an elevation view, focus on facade elements and height dimensions.
    - If the drawing is a cropped block, extract relevant information from each block.
    - For unit details, include bedroom and bathroom counts if visible or can be inferred.
    - Attempt to differentiate units if you suspect multiple types with different bedroom or bathroom counts.
"""

questions = """
        What is the drawing number of this Drawing Title "분산상가-1 지하3층 평면도 (근린생활시설-3)"?
        What is the project name of this Drawing Title ""?
        What is the revision of this Drawing Title ""?
        What is the scale of this Drawing Title ""?
        What is the architect name of this Drawing Title ""?
        What is the notes of this Drawing Title ""?
        What is the table of this Drawing Title ""?
        """