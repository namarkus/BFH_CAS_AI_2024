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

class InputFileHandler:

    def __init__(self, mode: str, config: VbcConfig):
        self.mode = mode
        self.input_path = config.sources_path
        self.repostitory_path = config.knowledge_repository_path
        self.logger = app_logger()

    def __filter_files(self, directory, extensions):
        filtered_files = []
        for ext in extensions:
            filtered_files.extend(glob.glob(f"{directory}/*{ext}"))
        return filtered_files

    def get_input_files(self):
        filtered_input_files = self.__filter_files(self.input_path, SUPPORTED_INPUT_FILE_TYPES)
        self.logger.debug(f"Habe {len(filtered_input_files)} akzeptierte Dateien in {self.input_path} gefunden.")
        if (self.mode == "inc"):
            self.logger.debug("Filtere Dateien für inkrementelle Verarbeitung...")
            processed_files = self.__filter_files(self.repostitory_path, SUPPORTED_REPO_FILE_TYPES)
            self.logger.info(f"Das Verzeichnis {self.repostitory_path} beinhaltet {len(processed_files)} verarbeitete Dateien.")
            unprocessed_input_files = [f for f in filtered_input_files if f not in processed_files] #fixme ... hier muss noch die Dateiendung entfernt werden; Alternativ Inhalt auf Version prüfen 
            self.logger.debug(f"Nach dem Filtern sind noch {len(unprocessed_input_files)} neue, unverarbeitete Dateien vorhanden.")
            filtered_input_files = unprocessed_input_files
        return [InputFile(file) for file in filtered_input_files]

            
class InputFile:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = file_path.split("/")[-1]
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
        data_uri = f"data:image/png;base64,{base64_png}"
        return data_uri

class MetaFile:
    def __init__(self, config: VbcConfig, from_json_file:str=None, from_input_file:InputFile=None):
        if from_input_file is not None:
            self.path = f"{config.knowledge_repository_path}/{from_input_file.file_name}.json"
            self.metadata = {
                "input_file": from_input_file.file_name,
                "processed_at": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                "processor": {
                    "machine": {
                        "name": platform.node(),
                        "architecture": platform.machine(),
                        "details": platform.uname(),
                        "os": {
                            "name": platform.system(),
                            "version": platform.release(),
                        },
                    },
                    "app" : {
                        "name" : "vbc_learn" ,
                        "version" : "__version__",
                    },
                    "user": getpass.getuser()
                },
                "pages": [],
                "embeddings": []
            }
        elif from_json_file is not None:
            selt._file_path = from_json_file
            self.metadata = _load()
        else:    
            raise Exception("Es wurde keine Datei zum Laden angegeben.")


    def __load(self):
        with open(self.file_path, 'r') as f:
            self.data = json.metadata(f)
        
    def save(self):
        with open(json_path, 'w') as f:
            json.dump(self.metadata, f , indent=2)
    
    

