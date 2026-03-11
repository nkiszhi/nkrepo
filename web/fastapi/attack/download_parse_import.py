#!/usr/bin/env python3
"""
Downloads MITRE ATT&CK Enterprise STIX bundle from GitHub,
parses it using stix2 library, and generates JSON files similar
to the Go implementation in mitre_parser_service.go.
"""

import json
import os
import requests
import stix2


class AttackDataProcessor:
    """MITRE ATT&CK STIX data processor."""
    
    def __init__(self, url=None, temp_filename="enterprise-attack.json", output_dir="parsed_output"):
        """Initialize processor with configuration."""
        self.url = url or "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
        self.temp_filename = temp_filename
        self.output_dir = output_dir
        self.tactics_map = {}
        self.techniques_map = {}
        self.translations = {}
        self.tactic_order = []
    
    def _check_local_file_exists(self, local_path):
        """Checks if the specified local file exists."""
        return os.path.exists(local_path)
    
    def _download_file(self, url, local_path):
        """Downloads a file from a URL to a local path."""
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded to {local_path}")
    
    def _get_external_id(self, stix_obj):
        """Extracts the MITRE ATT&CK external ID from a STIX object."""
        external_refs = self._get_attr(stix_obj, 'external_references')
        if external_refs:
            for ref in external_refs:
                if isinstance(ref, dict):
                    if ref.get('source_name') == 'mitre-attack':
                        return ref.get('external_id')
                elif hasattr(ref, 'source_name') and ref.source_name == 'mitre-attack':
                    return getattr(ref, 'external_id', None)
        return None
    
    def _get_mitre_url(self, stix_obj):
        """Extracts the MITRE ATT&CK URL from a STIX object."""
        external_refs = self._get_attr(stix_obj, 'external_references')
        if external_refs:
            for ref in external_refs:
                if isinstance(ref, dict):
                    if ref.get('source_name') == 'mitre-attack' and ref.get('url'):
                        return ref.get('url')
                elif hasattr(ref, 'source_name') and ref.source_name == 'mitre-attack' and hasattr(ref, 'url'):
                    return ref.url
        return ''
    
    def _get_attr(self, obj, attr_name):
        """Helper function to get attribute from both STIX objects and dicts."""
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        elif isinstance(obj, dict):
            return obj.get(attr_name)
        return None
    
    def _format_date(self, date_obj):
        """Format date object to ISO string."""
        if date_obj:
            if hasattr(date_obj, 'isoformat'):
                return date_obj.isoformat()
            elif isinstance(date_obj, str):
                return date_obj
        return ''
    
    def _get_kill_chain_phases(self, stix_obj):
        """Extracts kill chain phases for MITRE ATT&CK."""
        phases = []
        if hasattr(stix_obj, 'kill_chain_phases'):
            for phase in stix_obj.kill_chain_phases:
                if hasattr(phase, 'kill_chain_name') and phase.kill_chain_name == 'mitre-attack':
                    phases.append(phase.phase_name)
        elif isinstance(stix_obj, dict) and 'kill_chain_phases' in stix_obj:
            for phase in stix_obj['kill_chain_phases']:
                if isinstance(phase, dict) and phase.get('kill_chain_name') == 'mitre-attack':
                    phases.append(phase.get('phase_name'))
        return phases
    
    def _filter_revoked_deprecated(self, all_objects):
        """Filters out revoked and deprecated objects, collecting their info."""
        filtered_objects = []
        deprecated_objects = []
        
        # Convert list to dict for faster lookup when checking revoked-by relationships
        object_dict = {obj.id: obj for obj in all_objects if hasattr(obj, 'id')}
        
        for obj in all_objects:
            is_revoked = getattr(obj, 'revoked', False)
            is_deprecated = getattr(obj, 'x_mitre_deprecated', False)
            
            if is_revoked or is_deprecated:
                ext_id = self._get_external_id(obj)
                dep_info = {
                    'id': obj.id,
                    'type': obj.type,
                    'name': getattr(obj, 'name', ''),
                    'external_id': ext_id,
                    'revoked': is_revoked,
                    'deprecated': is_deprecated,
                }
                
                if is_revoked:
                    # Attempt to find the revoking object using a simple check
                    revoked_by_stix_id = getattr(obj, 'x_mitre_revoked_by', None)  # Common property used by MITRE
                    if not revoked_by_stix_id:
                        revoked_by_stix_id = getattr(obj, 'revoked_by', None)  # Standard STIX property
                    
                    if revoked_by_stix_id and revoked_by_stix_id in object_dict:
                        revoked_by_obj = object_dict[revoked_by_stix_id]
                        dep_info['revoked_by_id'] = self._get_external_id(revoked_by_obj) or revoked_by_obj.id  # Prefer external ID
                        dep_info['revoked_by_name'] = getattr(revoked_by_obj, 'name', '')
                        dep_info['reason'] = f"Revoked by {dep_info['revoked_by_name']} ({dep_info['revoked_by_id']})"
                    else:
                        dep_info['reason'] = "Object was revoked"
                elif is_deprecated:
                    dep_info['reason'] = "Object is deprecated"
                
                deprecated_objects.append(dep_info)
            else:
                filtered_objects.append(obj)
        
        return filtered_objects, deprecated_objects
    
    def download_data(self):
        """Download STIX data if not available locally."""
        if self._check_local_file_exists(self.temp_filename):
            print(f"Local file '{self.temp_filename}' found. Skipping download.")
        else:
            print(f"Local file '{self.temp_filename}' not found.")
            self._download_file(self.url, self.temp_filename)
    
    def parse_stix_bundle(self):
        """Parse STIX bundle and return objects."""
        print(f"Parsing STIX bundle from {self.temp_filename}...")
        try:
            with open(self.temp_filename, 'r', encoding='utf-8') as f:
                bundle_data = json.load(f)
            bundle = stix2.parse(bundle_data, allow_custom=True)
        except Exception as e:
            print(f"Error parsing STIX bundle: {e}")
            return None
        
        if not hasattr(bundle, 'objects'):
            print("Error: Bundle does not contain 'objects'.")
            return None
        
        return bundle.objects
    
    def extract_objects(self, all_objects):
        """Extract tactics and techniques from STIX objects."""
        print("Extracting tactics and techniques...")
        
        for obj in all_objects:
            # Handle both parsed STIX objects and dict objects (for custom MITRE types)
            if hasattr(obj, 'type'):
                obj_type = obj.type
                obj_id = getattr(obj, 'id', 'No ID')
            elif isinstance(obj, dict):
                obj_type = obj.get('type')
                obj_id = obj.get('id', 'No ID')
            else:
                continue  # Skip non-STIX objects
            
            if obj_type == 'x-mitre-tactic':
                self._process_tactic(obj)
            elif obj_type == 'attack-pattern':
                self._process_technique(obj)
    
    def _process_tactic(self, obj):
        """Process a single tactic object."""
        external_id = self._get_external_id(obj)
        shortname = self._get_attr(obj, 'x_mitre_shortname')
        name = self._get_attr(obj, 'name')
        obj_id = self._get_attr(obj, 'id')
        
        # Only process objects with valid MITRE ATT&CK external IDs (TAxxx format) and shortname
        if external_id and external_id.startswith('TA') and shortname:
            self.tactics_map[external_id] = {
                'id': external_id,
                'name': name,
                'description': self._get_attr(obj, 'description'),
                'shortname': shortname,
                'techniques': []  # Will be populated later
            }
        else:
            print(f"Skipping invalid tactic: {obj_id} - {name} - ExtID: {external_id} - Shortname: {shortname}")
    
    def _process_technique(self, obj):
        """Process a single technique object."""
        external_id = self._get_external_id(obj)
        
        # Only process objects with valid MITRE ATT&CK external IDs (Txxx format)
        if external_id and external_id.startswith('T'):
            is_sub = bool(self._get_attr(obj, 'x_mitre_is_subtechnique'))
            platforms = self._get_attr(obj, 'x_mitre_platforms') or []
            if isinstance(platforms, list):
                platforms = [p for p in platforms if isinstance(p, str)]
            
            tech_info = {
                'id': external_id,
                'name': self._get_attr(obj, 'name') or '',
                'description': self._get_attr(obj, 'description') or '',
                'platforms': platforms,
                'url': self._get_mitre_url(obj),
                'created': self._format_date(self._get_attr(obj, 'created')),
                'modified': self._format_date(self._get_attr(obj, 'modified')),
                'is_sub_technique': is_sub,
                'parent_id': None,  # Will be set if it's a sub-technique
                'sub_techniques': [],  # Will be populated later
                'tactic_phases': self._get_kill_chain_phases(obj)
            }
            
            if is_sub:
                # Determine parent ID (e.g., T1055.001 -> T1055)
                parent_ext_id = '.'.join(external_id.split('.')[:1])
                tech_info['parent_id'] = parent_ext_id
            
            self.techniques_map[external_id] = tech_info
        else:
            obj_id = self._get_attr(obj, 'id')
            name = self._get_attr(obj, 'name')
            print(f"Skipping invalid technique: {obj_id} - {name} - ExtID: {external_id}")
    
    def build_relationships(self):
        """Build relationships between sub-techniques and tactics."""
        print("Building relationships...")
        print(f"Found {len(self.tactics_map)} tactics and {len(self.techniques_map)} techniques")
        
        # 1. Link sub-techniques to their parents
        for tech_id, tech_info in self.techniques_map.items():
            if tech_info['is_sub_technique'] and tech_info['parent_id']:
                parent_id = tech_info['parent_id']
                if parent_id in self.techniques_map:
                    self.techniques_map[parent_id]['sub_techniques'].append(tech_info)
        
        # 2. Link techniques to tactics based on kill_chain_phases
        shortname_to_tactic_id = {info['shortname']: tid for tid, info in self.tactics_map.items()}
        
        # Clear existing techniques from tactics
        for tid in self.tactics_map:
            self.tactics_map[tid]['techniques'] = []
        
        # Assign techniques to their respective tactics
        for tech_id, tech_info in self.techniques_map.items():
            # Only top-level techniques should be linked to tactics
            if not tech_info['is_sub_technique']:
                for phase_name in tech_info['tactic_phases']:
                    if phase_name in shortname_to_tactic_id:
                        tactic_id = shortname_to_tactic_id[phase_name]
                        if tactic_id in self.tactics_map:
                            self.tactics_map[tactic_id]['techniques'].append(tech_info)
    
    def load_translations(self, translations_file="tactic_translations.json"):
        """Load tactic translations for ordering and naming."""
        if os.path.exists(translations_file):
            with open(translations_file, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
            self.tactic_order = list(self.translations.keys())
            print(f"Loaded {len(self.tactic_order)} tactics from translation file")
        else:
            print(f"Warning: {translations_file} not found, using default order")
            self.tactic_order = sorted(self.tactics_map.keys())
    
    def generate_attack_matrix(self):
        """Generate attack matrix structure with proper ordering."""
        print("Generating attack matrix...")
        
        attack_matrix = {}
        for tid in self.tactic_order:
            if tid in self.tactics_map:
                tinfo = self.tactics_map[tid]
                
                # Use translations if available, otherwise fallback
                if tid in self.translations:
                    en_name = self.translations[tid]['en']
                    cn_name = self.translations[tid]['cn']
                else:
                    en_name = tinfo['name']
                    cn_name = tinfo['name']  # Fallback
                
                tactic_entry = {
                    'tactic_name_en': en_name,
                    'tactic_name_cn': cn_name,
                    'techniques': []
                }
                
                for tech in tinfo['techniques']:
                    tech_entry = {
                        'id': tech['id'],
                        'name': tech['name'],
                        'sub': []  # Populate sub-techniques for this technique
                    }
                    for subtech in tech['sub_techniques']:
                        tech_entry['sub'].append({
                            'id': subtech['id'],
                            'name': subtech['name']
                        })
                    tactic_entry['techniques'].append(tech_entry)
                
                attack_matrix[tid] = tactic_entry
            else:
                print(f"Warning: Tactic {tid} from translation file not found in parsed data")
        
        return attack_matrix
    
    def generate_technique_details(self):
        """Generate technique details structure."""
        print("Generating technique details...")
        
        technique_details = {'techniques': {}}
        for tid, tinfo in self.techniques_map.items():
            detail_info = {
                'name': tinfo['name'],
                'description': tinfo['description'],
                'platforms': tinfo['platforms'],
                'tactics': tinfo['tactic_phases'],
                'sub_techniques': [st['id'] for st in tinfo['sub_techniques']],  # Store only IDs
                'is_sub_technique': tinfo['is_sub_technique'],
                'parent_id': tinfo['parent_id'],
                'mitre_url': tinfo['url'],
                'created': tinfo['created'],
                'modified': tinfo['modified']
            }
            technique_details['techniques'][tid] = detail_info
        
        return technique_details
    
    def save_output_files(self, attack_matrix, technique_details, deprecated_objects):
        """Save all output files to specified directory."""
        print(f"Writing output files to '{self.output_dir}'...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save Attack Matrix
        matrix_path = os.path.join(self.output_dir, "attack-matrix.json")
        with open(matrix_path, 'w', encoding='utf-8') as f:
            json.dump(attack_matrix, f, indent=2, ensure_ascii=False)
        print(f"Saved attack matrix to {matrix_path}")
        
        # Save Technique Details
        details_path = os.path.join(self.output_dir, "technique-details.json")
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(technique_details, f, indent=2, ensure_ascii=False)
        print(f"Saved technique details to {details_path}")
        
        # Save Deprecated/Revoked Objects (if any)
        if deprecated_objects:
            deprecated_path = os.path.join(self.output_dir, "deprecated-objects.json")
            with open(deprecated_path, 'w', encoding='utf-8') as f:
                json.dump(deprecated_objects, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(deprecated_objects)} deprecated/revoked objects to {deprecated_path}")
        else:
            print("No deprecated or revoked objects found.")
    
    def process(self, translations_file="tactic_translations.json"):
        """Main processing pipeline."""
        # Download data if needed
        self.download_data()
        
        # Parse STIX bundle
        all_objects = self.parse_stix_bundle()
        if all_objects is None:
            return
        
        # Filter objects
        print("Filtering revoked and deprecated objects...")
        filtered_objects, deprecated_objects = self._filter_revoked_deprecated(all_objects)
        
        # Extract tactics and techniques
        self.extract_objects(filtered_objects)
        
        # Build relationships
        self.build_relationships()
        
        # Load translations
        self.load_translations(translations_file)
        
        # Generate output structures
        attack_matrix = self.generate_attack_matrix()
        technique_details = self.generate_technique_details()
        
        # Save output files
        self.save_output_files(attack_matrix, technique_details, deprecated_objects)


def main():
    """Main execution function."""
    processor = AttackDataProcessor()
    processor.process()


if __name__ == "__main__":
    main()