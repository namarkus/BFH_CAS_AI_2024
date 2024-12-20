#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _errors.py ]___________________________________________________________
"""
Exceptions, die speziell behandlet werden sollen.

Stellt als integrierter Bestandteil des Projektes "Versicherungsbedingungs 
Chat" Funktionen für das Fehlerhanlding bereit..
"""
if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________

class VbcConfigError(Exception):
    """
    Basisklasse für alle Exceptions, die im Projekt "Versicherungsbedingungs Chat" 
    geworfen werden.
    """
    def __init__(self, number: int, message: str, explanation:str):
        super().__init__(message)
        self.id = f"VBC-{number:03d}" 
        self.message = message
        self.explanation = explanation

    def __str__(self):
        return f"{self.id} - {self.message}"
