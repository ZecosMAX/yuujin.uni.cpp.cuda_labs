#include "zluda_utils.hpp"
#include "windows_utils.hpp"

#include <iostream>
#include <vector>
#include <codecvt>

using namespace std;

void RunWithZluda(int argc, char* argv[])
{
	auto directory = GetApplicationDirectory();
	auto appName = GetApplicationName();

	std::vector<std::string> args(argv, argv + argc);
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	std::wstring cmdLineArgs;

	cmdLineArgs += L" \"";
	cmdLineArgs += directory;
	cmdLineArgs += L"\\";
	cmdLineArgs += appName;
	cmdLineArgs += L"\"";
	cmdLineArgs += L" zluda";

	for (size_t i = 1; i < args.size(); i++)
	{
		cmdLineArgs += L" -";
		cmdLineArgs += converter.from_bytes(args[i]);
	}

	wcout << L"path: " << (directory + L"\\zluda\\zluda.exe") << endl;
	wcout << L"args: " << cmdLineArgs << endl;

	StartApplication(
		(LPWSTR)(const wchar_t*)((directory + L"\\zluda\\zluda.exe").data()),
		(LPWSTR)(const wchar_t*)(cmdLineArgs.data())
	);
}