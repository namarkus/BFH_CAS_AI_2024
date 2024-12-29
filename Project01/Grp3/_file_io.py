#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _configs.py ]__________________________________________________________
"""
Diverse Mechanismen für die Behanldung von Dateien im lokalen Dateisystem.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Funktionen für DateinI/O bereint.
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
import io
import os
import glob
import base64
import platform
import getpass
import json
from datetime import datetime
from _configs import VbcConfig    
from _logging import app_logger
from pdf2image import convert_from_path

SUPPORTED_INPUT_FILE_TYPES = ["*.pdf", "*.url"]
SUPPORTED_REPO_FILE_TYPES = ["*.json"]
            
class InputFile:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.file_extension = file_path.split(".")[-1]

    def __load(self):
        with open(self.file_path, "r") as file:
            return file.read()
    
    def is_processable_as_image(self):
        return self.file_extension in ["pdf"]

    def get_content(self):
        content = []
        if self.__is_pdf():
            page_images = convert_from_path(self.file_path)
            for page_image in page_images:
                content.append(self.__as_base64_uri(page_image))
        # elif self.__is_url():
        #     content.append(self.__load())
        else:
            file_content = self.__load()        
            content.append(file_content)
        return content

    def __is_pdf(self):
        return self.file_extension == "pdf"
    
    # def __is_url(self):    
    #     return self.file_extension == "url"
    
    def __as_base64_uri(self, img):
        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)
        base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')
        return base64_png

class MetaFile:
    def __init__(self, config: VbcConfig, 
                 from_json_file:str=None, from_input_file:InputFile=None):
        if from_input_file is not None:
            self.file_path = f"{config.knowledge_repository_path}/{from_input_file.file_name}.json"
            self.metadata = {
                "input_file": from_input_file.file_name,
                "processed_at": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                "profile": config.as_profile_label(),
                "processor": {
                    "app" : {
                        "name" : "vbc_learn" ,
                        "version" : config.learn_version ,
                    },
                    "text_preparation": {
                        "provider": config.image2text_llm_provider.value,
                    },
                    "embeddings": {
                        "provider": config.embeddings_provider.value,
                        "chunking_mode": config.chunking_mode.value,    
                        "storage": config.embeddings_storage.value,
                        "index_id": None
                    },                
                    "machine": {
                        "name": platform.node(),
                        "architecture": platform.machine(),
                        "os": {
                            "name": platform.system(),
                            "version": platform.release(),
                        },
                    },
                    "user": getpass.getuser()
                },
                "pages": [],
                "chunks": []
            }
        elif from_json_file is not None:
            self.file_path = from_json_file
            self.metadata = self.__load()
        else:    
            raise Exception("Es wurde keine Datei zum Laden angegeben.")

    def __load(self):
        with open(self.file_path, 'r') as f:
            return json.load(f)
        
    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.metadata, f , indent=2)
    
    def add_page(self, page):
        self.metadata["pages"].append(page)

    def get_pages(self) -> list[str]:
        return self.metadata["pages"]

    def add_chunk(self, chunk):
        self.metadata["chunks"].append(chunk)

    def remove_chunks(self):
        self.metadata["chunks"] = []

    def get_chunks(self) -> list[str]:
        return self.metadata["chunks"]

class InputFileHandler:

    def __init__(self, mode: str, config: VbcConfig):
        self.mode = mode
        self.input_path = config.sources_path
        self.repostitory_path = config.knowledge_repository_path
        self.config = config
        self.logger = app_logger()

    def __filter_files(self, directory, extensions):
        filtered_files = []
        for ext in extensions:
            filtered_files.extend(glob.glob(f"{directory}/*{ext}"))
        filtered_files.sort()
        return filtered_files

    def get_all_input_files(self) -> list[InputFile]:
        all_supported_input_files = self.__filter_files(self.input_path, SUPPORTED_INPUT_FILE_TYPES)
        self.logger.debug(f"Verzeichnis '{self.input_path}' enthält {len(all_supported_input_files)} Dateien mit unterstütztem Format.")
        return [InputFile(file) for file in all_supported_input_files]
        
    def get_input_files_to_process(self, config: VbcConfig) -> list[InputFile]:
        all_input_files = self.get_all_input_files()
        known_files = self.get_all_metafiles()
        for known_file in known_files:
            if self.__is_already_text_prepared(known_file, config):
                self.logger.debug(f"Datei {known_file.metadata['input_file']} wurde bereits verarbeitet.")
                all_input_files = [input_file for input_file in all_input_files if input_file.file_name != known_file.metadata["input_file"]]
        return all_input_files

    def delete_all_metafiles(self):
        meta_files = self.__filter_files(self.repostitory_path, SUPPORTED_REPO_FILE_TYPES)
        for file in meta_files:
            self.logger.debug(f"Lösche Datei {file}")
            os.remove(file)

    def get_all_metafiles(self) -> list[MetaFile]:
        filtered_meta_files = self.__filter_files(self.repostitory_path, SUPPORTED_REPO_FILE_TYPES)
        self.logger.debug(f"Repository '{self.repostitory_path}' enthält {len(filtered_meta_files)} bekannte Dokumente.")
        return [MetaFile(self.config, from_json_file=file) for file in filtered_meta_files]
    
    def get_metafiles_to_chunk(self, config: VbcConfig) -> list[MetaFile]:
        all_meta_files = self.get_all_metafiles()
        dirty_files = []
        for meta_file in all_meta_files:
            if not self.__is_already_chunked(meta_file, config):
                self.logger.debug(f"Datei {meta_file.file_path} wird erneut verarbeitet.")
                dirty_files.append(meta_file)
        return dirty_files
    
    def get_metafiles_to_embed(self, config: VbcConfig) -> list[MetaFile]:
        all_meta_files = self.get_all_metafiles()
        dirty_files = []
        for meta_file in all_meta_files:
            if not self.__is_already_embedded(meta_file, config):
                self.logger.debug(f"Datei {meta_file.file_path} wird erneut verarbeitet.")
                dirty_files.append(meta_file)
        return dirty_files

    def __is_already_text_prepared(self, meta_file: MetaFile, config: VbcConfig):
        return (meta_file.metadata["processor"]["text_preparation"]["provider"] == config.image2text_llm_provider.value 
                and meta_file.metadata["processor"]["app"]["version"] == config.learn_version)
    
    def __is_already_chunked(self, meta_file: MetaFile, config: VbcConfig):
        return (len(meta_file.metadata["chunks"]) > 0
                and meta_file.metadata["processor"]["embeddings"]["chunking_mode"] == config.chunking_mode.value                
                and meta_file.metadata["processor"]["app"]["version"] == config.learn_version)

    def __is_already_embedded(self, meta_file: MetaFile, config: VbcConfig):
        return (meta_file.metadata["processor"]["embeddings"]["index_id"] is not None
                and meta_file.metadata["processor"]["embeddings"]["storage"] == config.embeddings_storage.value
                and meta_file.metadata["processor"]["embeddings"]["provider"] == config.embeddings_provider.value
                and meta_file.metadata["processor"]["embeddings"]["chunking_mode"] == config.chunking_mode.value                
                and meta_file.metadata["processor"]["app"]["version"] == config.learn_version)
