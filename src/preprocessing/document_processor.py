"""
Module xử lý và tiền xử lý văn bản pháp luật cho hệ thống RAG UIT@PubHealthQA.
"""

import re
import logging
import unicodedata
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any

def parse_document_id(ten_van_ban: str, noi_dung: str) -> str:
    """
    Trích xuất số hiệu/mã định danh chuẩn của văn bản.
    
    Args:
        ten_van_ban: Tên của văn bản
        noi_dung: Nội dung văn bản
        
    Returns:
        Mã định danh của văn bản
    """
    # Phân tích từ TenVanBan
    match = re.search(r"(\d+[\w/.-]+(?:TT-BYT|NĐ-CP|QH\d+|Lệnh|QĐ-BYT|BGDĐT|BTC|BLĐTBXH|BTTTT|BTNMT|BTP|BVHTTDL|BYT|BCT|BNN&PTNT|BKH&ĐT|BCA|BQP|BNG|BNV|CP|UBTVQH\d+|VPCP|TTg|CT|NQ-CP|\w+))", ten_van_ban)
    if match:
        return match.group(1)

    # Tìm trong nội dung
    match_nd = re.search(r"Số:\s*([\w/.-]+(?:TT-BYT|NĐ-CP|QH\d+|Lệnh|QĐ-BYT|BGDĐT|BTC|BLĐTBXH|BTTTT|BTNMT|BTP|BVHTTDL|BYT|BCT|BNN&PTNT|BKH&ĐT|BCA|BQP|BNG|BNV|CP|UBTVQH\d+|VPCP|TTg|CT|NQ-CP|\w+))", noi_dung, re.IGNORECASE)
    if match_nd:
        return match_nd.group(1)

    # Trích xuất tên văn bản, bỏ loại văn bản
    cleaned_name = re.sub(r"^(Thông tư|Nghị định|Luật|Quyết định|Công văn|Lệnh|Nghị quyết)\s+", "", ten_van_ban, flags=re.IGNORECASE).strip()
    return cleaned_name if cleaned_name else f"UnknownID_{ten_van_ban[:20]}"

def parse_effective_date(date_str: str) -> Optional[str]:
    """
    Phân tích chuỗi ngày dạng DD/MM/YYYY sang YYYY-MM-DD.
    
    Args:
        date_str: Chuỗi ngày theo định dạng DD/MM/YYYY
        
    Returns:
        Chuỗi ngày theo định dạng YYYY-MM-DD, hoặc None nếu lỗi
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    try:
        # Xử lý trường hợp ngày/tháng chỉ có 1 chữ số
        parts = date_str.split('/')
        if len(parts) == 3:
            day = parts[0].zfill(2)
            month = parts[1].zfill(2)
            year = parts[2]
            
            current_year = datetime.now().year
            if not year.isdigit() or int(year) < 1900 or int(year) > current_year + 10:
                raise ValueError(f"Năm không hợp lệ: {year}")
            
            if not month.isdigit() or not (1 <= int(month) <= 12):
                raise ValueError(f"Tháng không hợp lệ: {month}")
            
            if not day.isdigit() or not (1 <= int(day) <= 31):
                raise ValueError(f"Ngày không hợp lệ: {day}")
            
            datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y")
            return f"{year}-{month}-{day}"
        else:
            logging.warning(f"Định dạng ngày không đúng DD/MM/YYYY: {date_str}")
            return date_str  # Trả về gốc nếu không đúng định dạng
    except ValueError as e:
        logging.warning(f"Không thể phân tích ngày '{date_str}': {e}")
        return date_str  # Trả về gốc nếu lỗi

def clean_content(text: str) -> Tuple[str, int, int]:
    """
    Làm sạch nội dung văn bản, trả về nội dung chính và vị trí bắt đầu/kết thúc.
    
    Args:
        text: Nội dung văn bản cần làm sạch
        
    Returns:
        Tuple gồm (nội_dung_đã_làm_sạch, vị_trí_bắt_đầu, vị_trí_kết_thúc)
    """
    start_index = 0
    end_index = len(text)
    lines = text.splitlines()
    if not lines:
        return "", 0, 0

    found_start = False
    start_content_markers = [
        r"^\s*Chương\s+[IVXLCDM]+", r"^\s*Phần\s+(?:thứ\s+)?\w+", r"^\s*Điều\s+1\b"
    ]
    title_keywords = ["QUY ĐỊNH", "HƯỚNG DẪN", "SỬA ĐỔI", "BỔ SUNG", "BAN HÀNH"]
    title_line_index = -1
    
    # Tìm dòng tiêu đề chính
    for i, line in enumerate(lines[:min(len(lines), 30)]):
        if re.match(r"^(BỘ|ỦY BAN|CHÍNH PHỦ|VĂN PHÒNG|CỘNG HÒA|Độc lập)", line.strip(), re.IGNORECASE):
            continue
        if any(keyword in line.upper() for keyword in title_keywords):
            if i > 0 and (re.search(r"(?:Hà Nội|TP\. Hồ Chí Minh|.*),\s*ngày\s+\d+\s+tháng\s+\d+\s+năm\s+\d+", lines[i-1]) or re.search(r"Số:\s*\d+", lines[i-1])):
                title_line_index = i
                break

    # Tìm kết thúc phần căn cứ
    citation_end_index = -1
    search_limit = min(len(lines), 60)
    start_search_citation = title_line_index + 1 if title_line_index != -1 else 0

    for i in range(start_search_citation, search_limit):
        line_strip = lines[i].strip()
        if line_strip.lower().startswith("căn cứ"):
            citation_end_index = i
        # Hoặc dòng cuối cùng kết thúc bằng ; trong khối căn cứ
        elif line_strip.endswith(";") and citation_end_index >= start_search_citation:
            citation_end_index = i
        # Dừng nếu gặp dòng không phải căn cứ, không phải đề nghị, và không rỗng (sau khi đã thấy căn cứ)
        elif citation_end_index != -1 and line_strip and not line_strip.lower().startswith("theo đề nghị"):
            break
        # Dừng nếu gặp dòng "theo đề nghị"
        elif line_strip.lower().startswith("theo đề nghị"):
            # Nếu dòng này nằm ngay sau căn cứ cuối cùng, coi nó là hết căn cứ
            if i == citation_end_index + 1:
                citation_end_index = i
            break  # Luôn dừng khi gặp "theo đề nghị"

    content_start_line_index = citation_end_index + 1
    # Bỏ qua các dòng trống hoặc "Bộ trưởng...", "Theo đề nghị..." sau căn cứ
    while content_start_line_index < len(lines):
        line_strip = lines[content_start_line_index].strip()
        if not line_strip:  # Dòng trống
            content_start_line_index += 1
            continue
        # Các chức danh/cụm từ phổ biến cần bỏ qua
        if line_strip.lower().startswith(("bộ trưởng", "thủ tướng chính phủ", "theo đề nghị", "chủ tịch")):
            content_start_line_index += 1
            continue
        # Nếu dòng hiện tại là cấu trúc hoặc có nội dung -> dừng bỏ qua
        if any(re.match(p, line_strip, re.IGNORECASE) for p in start_content_markers) or line_strip:
            found_start = True
            break
        # Nếu không phải các trường hợp trên, vẫn tiếp tục bỏ qua (có thể là tên người đề nghị)
        content_start_line_index += 1

    # Tính vị trí bắt đầu
    if found_start:
        start_index = sum(len(l) + 1 for l in lines[:content_start_line_index])
    else:  # Fallback nếu không tìm thấy điểm bắt đầu rõ ràng
        start_index = 0
        logging.debug(f"Không thể xác định rõ điểm bắt đầu nội dung cho VB: {text[:150]}...")

    # --- Tìm vị trí kết thúc nội dung chính ---
    end_patterns = [
        # Ưu tiên các marker kết thúc rõ ràng
        r"^\s*PHỤ LỤC\s*\d*", r"^\s*DANH MỤC KÈM THEO", r"^\s*Biểu\s*\d+", r"^\s*Mẫu số\s*\d+",
        r"^\s*Nơi nhận:",
        # Chữ ký (KT., Chức danh)
        r"^\s*KT\.?\s*(?:BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|TỔNG GIÁM ĐỐC|GIÁM ĐỐC)",
        r"^\s*(?:THỨ TRƯỞNG|PHÓ THỦ TƯỚNG|PHÓ CHỦ TỊCH|BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|QUYỀN BỘ TRƯỞNG|CHỦ NHIỆM|TỔNG GIÁM ĐỐC|GIÁM ĐỐC)\b",
        r"^\s*([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲÝỴỶỸĐ]+(\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲÝỴỶỸĐ]+)*)$",
    ]
    end_line_index = len(lines)
    search_start_rev = max(0, len(lines) - 100)
    if start_index > 0:
        start_line_approx = len(text[:start_index].splitlines())
        search_start_rev = max(search_start_rev, start_line_approx)

    found_end_marker = False
    # Tìm từ dưới lên
    for i in range(len(lines) - 1, search_start_rev - 1, -1):
        line_strip = lines[i].strip()
        # Bỏ qua dòng quá dài hoặc quá ngắn không thể là marker
        if len(line_strip) > 100 or len(line_strip) < 3:
            continue

        # Kiểm tra các mẫu kết thúc
        if any(re.match(p, line_strip, re.IGNORECASE) for p in end_patterns):
            end_line_index = i
            found_end_marker = True
            logging.debug(f"Tìm thấy marker kết thúc '{line_strip}' tại dòng {i}")
            break  # Dừng ngay khi thấy marker mạnh

    # Nếu tìm thấy marker, lùi lên bỏ qua các dòng trống hoặc tên người ký/chức danh ngay trên đó
    if found_end_marker:
        temp_idx = end_line_index - 1
        while temp_idx >= 0:
            prev_line_strip = lines[temp_idx].strip()
            is_empty = not prev_line_strip
            # Chức danh thường viết hoa, ít từ
            is_title = prev_line_strip.isupper() and len(prev_line_strip.split()) < 7
            # Tên người ký: heuristic đơn giản (viết hoa chữ cái đầu, có khoảng trắng)
            is_signer_name = bool(re.match(r"^[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+$", prev_line_strip))

            if is_empty or is_title or is_signer_name:
                end_line_index = temp_idx  # Cập nhật dòng kết thúc thực sự
                temp_idx -= 1
            else:
                # Dừng khi gặp dòng không phải là trống/chức danh/tên
                break

    # Tính vị trí ký tự kết thúc
    if end_line_index < len(lines):
        end_index = sum(len(l) + 1 for l in lines[:end_line_index])
        end_index = max(0, end_index - 1) if end_index > 0 else 0  # Trừ newline cuối
    else:
        end_index = len(text)

    # Đảm bảo end >= start và nội dung không quá ngắn
    end_index = max(start_index, end_index)
    main_content = text[start_index:end_index].strip()
    if len(main_content) < 50:  # Nếu nội dung chính quá ngắn sau khi cắt -> có thể lỗi
        logging.warning(f"Nội dung chính quá ngắn ({len(main_content)} chars) sau khi làm sạch. Kiểm tra lại VB: {text[:150]}...")

    # Xóa các dòng chỉ chứa số trang ở cuối/đầu nếu có
    main_content = re.sub(r'\n\s*\d+\s*$', '', main_content).strip()
    main_content = re.sub(r'^\s*\d+\s*\n', '', main_content).strip()

    return main_content, start_index, end_index

def find_structural_elements(text: str) -> List[Dict[str, Any]]:
    """
    Tìm tất cả các yếu tố cấu trúc (Chương, Mục, Điều, Khoản, Điểm) và vị trí của chúng.
    
    Args:
        text: Nội dung văn bản
        
    Returns:
        Danh sách các yếu tố cấu trúc đã tìm thấy
    """
    elements = []
    # Pattern được tối ưu hơn để bắt tiêu đề chính xác hơn
    patterns = {
        # Chương/Mục/Điều: Bắt số hiệu (la mã/thường), dấu chấm tùy chọn, sau đó là tiêu đề
        'Chương': r"^\s*Chương\s+([IVXLCDM\d]+)\b\.?\s*(.*?)(?=\n\s*(?:Chương|Mục|Điều|\d+\.|[a-zđ]\))|$)",
        'Mục': r"^\s*Mục\s+(\d+|[IVXLCDM]+)\b\.?\s*(.*?)(?=\n\s*(?:Chương|Mục|Điều|\d+\.|[a-zđ]\))|$)",
        'Điều': r"^\s*Điều\s+(\d+)\b\.?\s*(.*?)(?=\n\s*(?:Chương|Mục|Điều|\d+\.|[a-zđ]\))|$)",
        'Khoản': r"^\s*(\d+)\.\s+",
        'Điểm': r"^\s*([a-zđ])\)\s+",
    }

    for type, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            identifier = match.group(1).strip()
            # Lấy tiêu đề nếu có group 2
            title = match.group(2).strip().replace('\n', ' ') if len(match.groups()) > 1 else ""
            # Chỉ lấy title nếu nó không quá dài
            if title and len(title) >= 250:  # Tăng giới hạn độ dài title
                title = ""

            elements.append({
                'type': type,
                'identifier': identifier,
                'title': title,
                'start': match.start(),
                'end': match.end()
            })

    # Sắp xếp các phần tử theo vị trí bắt đầu
    elements.sort(key=lambda x: x['start'])
    return elements

def get_contextual_structure(char_index: int, sorted_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tìm ngữ cảnh cấu trúc (Chương->Mục->Điều->Khoản->Điểm) cho một vị trí ký tự cụ thể.
    
    Args:
        char_index: Vị trí ký tự cần xác định ngữ cảnh
        sorted_elements: Danh sách các yếu tố cấu trúc đã tìm thấy và sắp xếp theo vị trí
        
    Returns:
        Từ điển chứa thông tin ngữ cảnh cấu trúc
    """
    context = {}
    last_element_of_type = {}
    hierarchy = ['Chương', 'Mục', 'Điều', 'Khoản', 'Điểm']

    current_context = {}
    for element in sorted_elements:
        if element['start'] <= char_index:
            current_context[element['type']] = {
                'identifier': element['identifier'],
                'title': element.get('title', '')
            }
            current_level_index = hierarchy.index(element['type'])
            # Xóa ngữ cảnh cấp thấp hơn khi gặp cấp cao hơn
            to_delete = []
            for existing_type in current_context:
                if existing_type != element['type'] and hierarchy.index(existing_type) > current_level_index:
                    to_delete.append(existing_type)
            for key_to_del in to_delete:
                del current_context[key_to_del]

            last_element_of_type = current_context.copy()
        else:
            break

    location_parts = []
    context_dict = {}
    for struct_type in hierarchy:
        if struct_type in last_element_of_type:
            info = last_element_of_type[struct_type]
            part = f"{struct_type} {info['identifier']}"
            # Chỉ thêm title cho Chương, Mục, Điều
            if struct_type in ['Chương', 'Mục', 'Điều'] and info['title'] and len(info['title']) < 150:
                part += f": {info['title'][:100]}..."  # Rút gọn title nếu quá dài
            location_parts.append(part)
            context_dict[struct_type] = info['identifier']

    context_dict['location_detail'] = " -> ".join(location_parts) if location_parts else "Không có cấu trúc"
    return context_dict

def preprocess_text_for_embedding(text: str) -> str:
    """
    Sửa các lỗi định dạng phổ biến trong text trước khi embedding.
    
    Args:
        text: Văn bản cần tiền xử lý
        
    Returns:
        Văn bản đã được tiền xử lý
    """
    if not text:
        return ""

    # 1. Xử lý chuỗi dấu chấm ASCII liền nhau (3+)
    text = re.sub(r'\.{3,}', ' ', text)
    # 2. Xử lý ký tự dấu ba chấm Unicode (…) liền nhau (1+)
    text = re.sub(r'…+', ' ', text)
    # 3.Xử lý chuỗi dấu chấm ASCII có thể có khoảng trắng xen kẽ (3+ dấu chấm)
    text = re.sub(r'(\.\s*){3,}', ' ', text)

    # Thay thế chuỗi dài dấu gạch nối bằng khoảng trắng
    text = re.sub(r'-{3,}', ' ', text)

    # Sửa lỗi dính chữ cụ thể
    text = text.replace("hoặcc)", "hoặc c)")
    text = text.replace("thi hành1.", "thi hành 1.")
    text = text.replace("năm 2025.2.", "năm 2025. 2.")

    # Thêm khoảng trắng sau dấu câu nếu thiếu
    text = re.sub(r'([.;,:?)])([a-zA-Z0-9À-ỹ])', r'\1 \2', text)

    # Thêm khoảng trắng trước số/chữ cái đầu mục nếu dính liền chữ
    text = re.sub(r'([a-zA-ZÀ-ỹ])(\d+\.|[a-zđ]\))', r'\1 \2', text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    # Chuẩn hóa Unicode về dạng NFKC
    text = unicodedata.normalize('NFKC', text)

    # Loại bỏ ký tự lạ (Whitelist)
    allowed_chars = r'\w\s\d.,;:?!%()-/+ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲÝỴỶỸửữựỳýỵỷỹ'
    text = re.sub(f'[^{allowed_chars}]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()  # Dọn dẹp khoảng trắng lại

    return text