#include "errors.h"

std::string hResultToEnglish(HRESULT hr)
{
    char buf[8192] = { 0 };
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, DWORD(hr),
        MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
        buf,
        sizeof(buf),
        NULL);
    return std::string(buf);
}
void DX_THROW_ON_FAIL(HRESULT hr, std::string message, ID3D11Device* device, std::source_location location)
{
    if (!FAILED(hr)) return;

    std::ostringstream ss;
    ss << "DX_THROW_ON_FAIL has detected an error.\n";
    ss << "File " << location.file_name() << ", line " << location.line() << ", char " << location.column() << "\n"
        << "Function " << location.function_name() << "\n";
    if (!message.empty()) ss << "Message: " << message << "\n";

    ss << "HRESULT: 0x" << std::hex << hr << "\n";
    ss << hResultToEnglish(hr);
    if (device)
    {
        HRESULT removeReason = device->GetDeviceRemovedReason();
        if (FAILED(removeReason)) ss << "Device removed reason: " << hResultToEnglish(removeReason);
    }
    throw std::runtime_error(ss.str());
}