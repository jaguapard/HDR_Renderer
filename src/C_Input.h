#pragma once
#include <SDL3\SDL.h>
#include <unordered_map>
#include <unordered_set>

class C_Input
{
public:
	C_Input(const C_Input&) = delete;
	C_Input(C_Input&&) = delete;
	C_Input& operator=(const C_Input&) = delete;
	C_Input& operator=(C_Input&&) = delete;

	//This should be called on beginning of every frame. All "xOnThisFrame" methods consider events processed by C_Input in the time from last beginNewFrame() call and corresponding method call
	void beginNewFrame();

	//Take an event and process it. This should be called on every frame for each event. Any events not pretaining to input are safely ignored
	void handleEvent(const SDL_Event& ev);

	bool isButtonHeld(SDL_Scancode k);

	//Returns true only on the frame the button was pressed on, holding it will not return true repeatedly.
	bool wasButtonPressedOnThisFrame(SDL_Scancode k);

	//Returns true only on the frame the button was pressed on, holding it will not return true repeatedly.
	bool wasCharPressedOnThisFrame(char c);

	bool isMouseButtonHeld(int button);
	static C_Input& getInstance();
private:
	C_Input();
	std::unordered_map<char, SDL_Scancode> charToScancodeMap;
	std::unordered_map<SDL_Scancode, char> scancodeToCharMap;

	std::unordered_set<SDL_Scancode> lockedKeys;

	std::unordered_map<SDL_Scancode, bool> buttonHoldStatus;
	std::unordered_map<SDL_Scancode, bool> buttonPressStatus;
	std::unordered_map<int, bool> mouseButtonHoldStatus;
};