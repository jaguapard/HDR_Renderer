#pragma once
#include <assimp/Importer.hpp>
#include <vector>

class AssetLoader
{
public:
	//WARNING: this structs aren't supposed to be used in renderers, and act just as easy wrappings, renderers should perform their optimizations (like indexed triangle lists) on their own!
	struct ImportedTriangle
	{
		float vertices[3][3]; //vertice vertices[i][j] = j'th vertice of triangle i
		float u[3], v[3];
	};
	struct ImportedModel
	{
		std::vector<ImportedTriangle> triangles;
		std::optional<std::string> diffuseMapPath, normalMapPath, metallicityPath, roughnessPath;
	};

	AssetLoader();
	std::vector<ImportedModel> loadObj(std::string path, std::string convertToSavePath = "");
	static std::vector<ImportedModel> loadBmdl(std::string path);

	//void saveModels(const std::vector<Model>& models, std::string path) const;

	
private:
	Assimp::Importer importer;
};