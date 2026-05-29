#pragma once
#include <string>
#include <sstream>
#include <optional>
#include <source_location>
#include <Windows.h>
#include <D3D11.h>

static void __raise_error_internal(const char* filePath, int line, std::string errorMsg, std::optional<HRESULT> code = std::nullopt)
{
    std::stringstream ss;
    ss << "Error in file: " << filePath << "\n" << "Line " << line << "\n" << errorMsg;
    if (code)
    {
        HRESULT hr = *code;
        char buf[8192] = { 0 };
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, DWORD(hr),
            MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
            buf,
            sizeof(buf),
            NULL);
        ss << "\nHRESULT: 0x" << std::hex << hr << "\n";
        ss << buf << "\n";
        /*
        if (hr == DXGI_ERROR_DEVICE_REMOVED)
        {
            ss << "DXGI device removed. Reason: " <<
        }*/
    }
    throw std::runtime_error(ss.str());
}
#define RAISE_ERROR(msg) (__raise_error_internal(__FILE__, __LINE__, std::string("Error: ")+msg))


void DX_THROW_ON_FAIL(HRESULT hr, std::string message = {}, ID3D11Device* device = nullptr, std::source_location location = std::source_location::current());