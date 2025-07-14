"""
Utilidades para manejo y sanitización de JSON
"""

import json
import re
import string
import logging

logger = logging.getLogger(__name__)

def robust_json_parse(json_string: str, fallback_value=None):
    """
    Intenta parsear JSON con múltiples estrategias de sanitización.
    
    Args:
        json_string: Cadena JSON a parsear
        fallback_value: Valor a retornar si todas las estrategias fallan
        
    Returns:
        Tuple: (parsed_data, success_flag, error_message)
    """
    if not json_string or not isinstance(json_string, str):
        return fallback_value, False, "Empty or invalid input"
    
    # Estrategia 1: Intentar parsear directamente
    try:
        return json.loads(json_string), True, None
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")
    
    # Estrategia 2: Sanitización básica
    try:
        sanitized = sanitize_json_string(json_string)
        return json.loads(sanitized), True, None
    except json.JSONDecodeError as e:
        logger.debug(f"Basic sanitization failed: {e}")
    
    # Estrategia 3: Sanitización agresiva
    try:
        aggressive_clean = ''.join(char for char in json_string if char in string.printable)
        return json.loads(aggressive_clean), True, None
    except json.JSONDecodeError as e:
        logger.debug(f"Aggressive sanitization failed: {e}")
    
    # Estrategia 4: Reparación de estructura
    try:
        aggressive_clean = ''.join(char for char in json_string if char in string.printable)
        fixed_json = aggressive_clean.rstrip(',').rstrip()
        
        # Intentar cerrar JSON incompleto
        if not fixed_json.endswith('}') and fixed_json.count('{') > fixed_json.count('}'):
            fixed_json += '}'
        
        return json.loads(fixed_json), True, None
    except json.JSONDecodeError as e:
        logger.debug(f"Structure repair failed: {e}")
    
    # Estrategia 5: Extracción de campos específicos con regex
    try:
        fields = extract_json_fields_with_regex(json_string)
        if fields:
            return fields, True, None
    except Exception as e:
        logger.debug(f"Regex extraction failed: {e}")
    
    return fallback_value, False, "All JSON parsing strategies failed"

def sanitize_json_string(json_string: str) -> str:
    """
    Sanitiza una cadena JSON eliminando caracteres de control inválidos.
    
    Args:
        json_string: Cadena JSON potencialmente con caracteres de control
        
    Returns:
        Cadena JSON sanitizada
    """
    # ASCII control characters (0-31) except \t(9), \n(10), \r(13)
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_string)
    
    # Unicode control characters
    sanitized = re.sub(r'[\u0080-\u009F]', '', sanitized)  # C1 control characters
    sanitized = re.sub(r'[\u2028\u2029]', '', sanitized)   # Line/Paragraph separators
    
    # Remove any remaining non-printable characters
    sanitized = re.sub(r'[^\x20-\x7E\t\n\r]', '', sanitized)
    
    return sanitized

def extract_json_fields_with_regex(json_string: str) -> dict:
    """
    Extrae campos JSON usando expresiones regulares como último recurso.
    
    Args:
        json_string: Cadena JSON malformada
        
    Returns:
        Diccionario con campos extraídos
    """
    fields = {}
    
    # Patrones para campos comunes
    patterns = {
        'answer': r'"answer"\s*:\s*"([^"]*)"',
        'confidence': r'"confidence"\s*:\s*([0-9.]+)',
        'completeness': r'"completeness"\s*:\s*"([^"]*)"',
        'key_points': r'"key_points"\s*:\s*\[(.*?)\]',
        'references_used': r'"references_used"\s*:\s*\[(.*?)\]'
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, json_string, re.DOTALL)
        if match:
            if field in ['confidence']:
                try:
                    fields[field] = float(match.group(1))
                except ValueError:
                    fields[field] = 0.5
            elif field in ['key_points', 'references_used']:
                # Extraer elementos de array
                array_content = match.group(1)
                if array_content.strip():
                    # Simplificado: split por comas y limpiar
                    items = [item.strip().strip('"') for item in array_content.split(',')]
                    fields[field] = [item for item in items if item]
                else:
                    fields[field] = []
            else:
                fields[field] = match.group(1)
    
    return fields

def safe_json_extract(response_object, field_name: str, default_value=None):
    """
    Extrae un campo de un objeto de respuesta de manera segura.
    
    Args:
        response_object: Objeto de respuesta que puede contener JSON
        field_name: Nombre del campo a extraer
        default_value: Valor por defecto si la extracción falla
        
    Returns:
        Valor del campo o valor por defecto
    """
    try:
        if hasattr(response_object, 'tool_calls') and response_object.tool_calls:
            tool_call = response_object.tool_calls[0]
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                parsed_data, success, error = robust_json_parse(tool_call.function.arguments)
                if success and isinstance(parsed_data, dict):
                    return parsed_data.get(field_name, default_value)
        
        # Fallback a contenido directo
        if hasattr(response_object, 'content') and response_object.content:
            return response_object.content
            
    except Exception as e:
        logger.warning(f"Safe JSON extract failed for field {field_name}: {e}")
    
    return default_value