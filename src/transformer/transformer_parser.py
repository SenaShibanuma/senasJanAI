# -*- coding: utf-8 -*-
import sys
import os
import gzip
import xml.etree.ElementTree as ET

def main():
    """
    【エラー分析・最終版】
    XML解析エラーが発生した際に、問題箇所周辺のテキストを詳細に出力し、
    根本原因の特定を目的とする。
    """
    data_dir = '/content/data'
    log_file_path = None
    
    print(f"Searching for a log file in '{data_dir}'...")
    if os.path.isdir(data_dir):
        try:
            log_file_path = next(entry.path for entry in os.scandir(data_dir) if entry.is_file())
        except StopIteration:
            pass
    
    if not log_file_path:
        print(f"Error: No log files found in '{data_dir}'.")
        return

    print(f"--- Analyzing single log file: {os.path.basename(log_file_path)} ---")
    
    log_text = ""
    try:
        # ファイル読み込み (gzip/通常ファイル両対応)
        try:
            with gzip.open(log_file_path, 'rb') as f: xml_content = f.read()
        except (gzip.BadGzipFile, OSError):
            with open(log_file_path, 'rb') as f: xml_content = f.read()
        
        log_text = xml_content.decode('utf-8').strip()
        print(" -> File reading and decoding successful.")

    except Exception as e:
        print(f"FATAL ERROR: Could not read or decode the file.")
        print(f" -> Reason: {e}")
        return

    # XML解析とエラー分析
    try:
        print("\nAttempting to parse XML...")
        # ログファイルは単一のルート<mjloggm>を持つはず
        root = ET.fromstring(log_text)
        print("\n--- SUCCESS! XML is well-formed and parsed correctly. ---")
        print("The issue likely lies within the game logic parser, not XML formatting.")

    except ET.ParseError as e:
        print(f"\n--- FATAL ERROR: The file content is not valid XML. ---")
        print(f" -> Reason: {e}")
        
        # エラー箇所を特定して表示
        # e.position は (line, column) のタプル
        line, col = e.position
        print(f" -> Error occurred at line {line}, column {col}.")
        
        # 複数行にまたがるログは稀なので、log_textをそのまま使う
        if col > 0 and len(log_text) > col:
            # 問題箇所の前後50文字を切り出して表示
            start = max(0, col - 50)
            end = min(len(log_text), col + 50)
            
            error_snippet = log_text[start:end]
            pointer_pos = col - start
            
            print("\n" + "="*20 + " ERROR CONTEXT " + "="*20)
            print(error_snippet)
            print(" " * pointer_pos + "^")
            print("="*55)
            print("Analysis: The character '^' points to the exact location where the parser failed.")

if __name__ == '__main__':
    main()