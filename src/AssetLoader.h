#pragma once
#include <assimp/Importer.hpp>
#include <vector>
#include <optional>
#include <memory>

class AssetLoader
{
public:
	#pragma pack(push, 1)
	struct ImportedVertex
	{
		aiVector3D space, normal;
		aiVector3D diffuseMapCoords, normalMapCoords;
	};
	//WARNING: this structs aren't supposed to be used in renderers, and act just as easy wrappings, renderers should perform optimizations (like indexed triangle lists, etc) on their own!
	struct ImportedTriangle
	{
		ImportedVertex v[3];
	};
	#pragma pack(pop)
	struct ImportedModel
	{
		std::vector<ImportedTriangle> triangles;
		std::optional<std::string> diffuseMapPath, normalMapPath, metallicityPath, roughnessPath;
	};

	AssetLoader();
	std::vector<ImportedModel> loadObj(std::string path, std::string convertToSavePath = "");
	static std::vector<ImportedModel> loadBmdl(std::string path); //compacted format for much faster and much less memory-consuming loading

	//void saveModels(const std::vector<Model>& models, std::string path) const;

	Assimp::Importer importer;
private:
	
};