#include "windows_utils.hpp"

using namespace std;

wstring GetApplicationDirectory()
{
	DWORD maxDirectorySize = 32760; // refer to https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-fscc/ffb795f3-027d-4a3c-997d-3085f2332f6f?redirectedfrom=MSDN 
	LPWSTR directoryString = new WCHAR[maxDirectorySize];
	//DWORD charsWritten = GetCurrentDirectoryW(maxDirectorySize, directoryString);
    DWORD charsWritten = GetModuleFileNameW(NULL, directoryString, maxDirectorySize);

	wstring modulePath(directoryString, charsWritten);
    int lastSlashPlace = modulePath.find_last_of(L'\\');

    wstring result(directoryString, lastSlashPlace);

	delete[] directoryString;

	return result;
}

wstring GetApplicationName()
{
    DWORD maxDirectorySize = 32760; // refer to https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-fscc/ffb795f3-027d-4a3c-997d-3085f2332f6f?redirectedfrom=MSDN 
    LPWSTR directoryString = new WCHAR[maxDirectorySize];
    //DWORD charsWritten = GetCurrentDirectoryW(maxDirectorySize, directoryString);
    DWORD charsWritten = GetModuleFileNameW(NULL, directoryString, maxDirectorySize);

    wstring modulePath(directoryString, charsWritten);
    int lastSlashPlace = modulePath.find_last_of(L'\\');

    wstring result(&directoryString[lastSlashPlace + 1], (charsWritten - lastSlashPlace - 1));

    delete[] directoryString;

    return result;
}

void StartApplication(LPWSTR lpApplicationName, LPWSTR lpCommandLine)
{
    // additional information
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;

    // set the size of the structures
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // start the program up
    CreateProcessW(lpApplicationName,   // the path
        lpCommandLine,  // Command line
        NULL,           // Process handle not inheritable
        NULL,           // Thread handle not inheritable
        FALSE,          // Set handle inheritance to FALSE
        0,              // No creation flags
        NULL,           // Use parent's environment block
        NULL,           // Use parent's starting directory 
        &si,            // Pointer to STARTUPINFO structure
        &pi             // Pointer to PROCESS_INFORMATION structure (removed extra parentheses)
    );

    // Close process and thread handles. 
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
}