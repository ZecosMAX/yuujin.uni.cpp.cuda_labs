#pragma once
#include <Windows.h>
#include <iostream>

std::wstring GetApplicationDirectory();
std::wstring GetApplicationName();
void StartApplication(LPWSTR lpApplicationName, LPWSTR lpCommandLine);