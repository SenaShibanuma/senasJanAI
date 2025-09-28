# -*- coding: utf-8 -*-
import sys
import os
import gzip
import xml.etree.ElementTree as ET
from pprint import pprint
# 本番のパーサーをインポートして、その動作を直接テストする
from .transformer_parser import TransformerParser

def main():
    """
    【エラー特定・最終版】
    本番用パーサーを直接呼び出し、どの段階で失敗しているかを詳細に報告する。
    """
    data_dir = '/content/data'
    log_file_path = None
    
    print(f"Searching for a log file in '{data_dir}'...")
    if os.path.isdir(data_dir):
        try:
            # 最初の1ファイルをデバッグ対象として取得
            log_file_path = next(entry.path for entry in os.scandir(data_dir) if entry.is_file())
        except StopIteration:
            pass
    
    if not log_file_path:
        print(f"Error: No log files found in '{data_dir}'.")
        return

    print(f"--- Analyzing single log file: {os.path.basename(log_file_path)} ---")
    
    # ---------------------------------------------------------------------
    # 本番パーサー(TransformerParser)の parse_log_file メソッドを、
    # エラーを隠さないように、ここで再実装してテストします。
    # ---------------------------------------------------------------------
    
    xml_content = None
    try:
        print("\nStep 1: Reading file content (trying gzip first)...")
        try:
            with gzip.open(log_file_path, 'rb') as f:
                xml_content = f.read()
            print(" -> Success: File was read as a gzip compressed file.")
        except (gzip.BadGzipFile, OSError):
            print(" -> Info: Not a gzip file. Trying as plain text...")
            with open(log_file_path, 'rb') as f:
                xml_content = f.read()
            print(" -> Success: File was read as a plain text file.")
    except Exception as e:
        print(f"\nFATAL ERROR in Step 1: Could not read the file at all.")
        print(f" -> Reason: {e}")
        return

    log_text = ""
    try:
        print("\nStep 2: Decoding file content from bytes to text (UTF-8)...")
        log_text = xml_content.decode('utf-8').strip().replace("</mjloggm>", "")
        print(" -> Success: Decoded content successfully.")
    except Exception as e:
        print(f"\nFATAL ERROR in Step 2: Could not decode the file content.")
        print(f" -> Reason: {e}")
        return

    try:
        print("\nStep 3: Parsing the text as XML...")
        # ET.fromstringはrootタグが1つである必要があるため、<root>で囲む
        root = ET.fromstring(f"<root>{log_text}</root>")
        print(f" -> Success: XML parsed successfully. Found {len(list(root))} top-level tags.")
    except Exception as e:
        print(f"\nFATAL ERROR in Step 3: The file content is not valid XML.")
        print(f" -> Reason: {e}")
        return

    # --- ここまでくれば、ファイル読み込みとXML解析は成功 ---
    
    print("\nStep 4: Running the full parsing logic...")
    parser = TransformerParser()
    training_data = parser.parse_log_file(log_file_path)

    if not training_data:
        print("\n--- RESULT: No training data was generated. ---")
        print("Analysis: File reading and XML parsing were successful, but the game logic in the parser did not find any valid training situations.")
        print("This is the final bug to solve.")
    else:
        print(f"\n--- SUCCESS! Generated {len(training_data)} training data points. ---")
        print("The parser is working correctly!")

if __name__ == '__main__':
    main()