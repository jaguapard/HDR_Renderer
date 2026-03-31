#include "AssetLoader.h"


#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "Vec.h"
#include <sstream>
#include <iostream>
#include <fstream>

#pragma comment(lib, "assimp-vc143-mt.lib")

Vec4f aiToBob(aiVector3D ai)
{
	return { ai.x, ai.y, ai.z, 0 };
}

std::string getFolderFromPath(std::string path, bool addTrailingSlash = false)
{
	size_t lastSlash = path.rfind('/');
	std::string pathStr;
	if (lastSlash != path.npos) pathStr = path.substr(0, lastSlash);
	else if (lastSlash != path.npos) pathStr = path.substr(0, lastSlash);
	else throw std::runtime_error("Could not extract folder from path string: " + path);

	if (addTrailingSlash) return pathStr + "/";
	else return pathStr;
}
AssetLoader::AssetLoader()
{

}

template <typename T>
void writeVarToFile(const T& var, std::ofstream& file, size_t sizeOverride = 0)
{
	if (!sizeOverride) file.write((const char*)(&var), sizeof(var));
	else file.write((const char*)(&var), sizeOverride);
}

template <typename T>
void readVarFromFile(T& var, std::ifstream& file, size_t sizeOverride = 0)
{
	if (!sizeOverride) file.read((char*)(&var), sizeof(var));
	else file.read((char*)(&var), sizeOverride);
}

std::vector<AssetLoader::ImportedModel> AssetLoader::loadObj(std::string path, std::string convertToSavePath)
{
	const auto pScene = this->importer.ReadFile(path.c_str(), aiProcess_Triangulate | aiProcess_PreTransformVertices | aiProcess_MakeLeftHanded | aiProcess_GenUVCoords);
	std::stringstream ss;
	ss << "Error while loading scene " << path << ": ";
	if (!pScene)
	{
		ss << this->importer.GetErrorString();
		throw std::runtime_error(ss.str());
	}

	std::vector<AssetLoader::ImportedModel> models;
	std::ofstream convertedSavedModel;
	if (convertToSavePath.length() > 0) convertedSavedModel = std::ofstream(convertToSavePath, std::ios::binary);

	for (size_t i = 0; i < pScene->mNumMeshes; i++)
	{
		aiMesh* mesh = pScene->mMeshes[i];
		aiMaterial* material = pScene->mMaterials[mesh->mMaterialIndex];
		aiString texturePath;
		material->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath);

		std::string textureRelPath = texturePath.C_Str();
		std::string textureFullPath = getFolderFromPath(path, true) + textureRelPath;

		std::vector<AssetLoader::ImportedTriangle> tris;
		for (size_t j = 0; j < mesh->mNumFaces; ++j)
		{
			aiFace face = mesh->mFaces[j];
			if (face.mNumIndices != 3)
			{
				ss << "Mesh " << i << " face " << j << " has unexpected vertice count: " << face.mNumIndices;
				throw std::runtime_error(ss.str());
			}

			AssetLoader::ImportedTriangle& t = tris.emplace_back();
			for (int k = 0; k < 3; ++k)
			{
				int vertIndex = face.mIndices[k];
				aiVector3D aiVertice = mesh->mVertices[vertIndex];
				aiVector3D aiUVs = mesh->mTextureCoords[0][vertIndex];
				aiUVs.y *= -1;

				auto v = aiToBob(aiVertice);
				t.vertices[k][0] = v[0];
				t.vertices[k][1] = v[1];
				t.vertices[k][2] = v[2];

				auto uv = aiToBob(aiUVs);
				t.u[k] = uv.x;
				t.v[k] = uv.x;
			}
		}

		ImportedModel model;
		model.diffuseMapPath = textureFullPath;
		model.triangles = tris;
		models.emplace_back(model);

		//if this is true, will save BMDL to path. This format is much lighter and fast to import
		if (convertToSavePath.length() > 0)
		{
			float saveData[15];
			uint64_t modelSize = tris.size() * sizeof(saveData); //size of a single model entry in bytes, not counting size member and path. 
			writeVarToFile(modelSize, convertedSavedModel);
			convertedSavedModel.write(textureRelPath.c_str(), textureRelPath.length() + 1);
			for (const auto& it : tris)
			{
				for (int k = 0; k < 3; ++k)
				{
					saveData[5 * k] = it.vertices[k][0];
					saveData[5 * k + 1] = it.vertices[k][1];
					saveData[5 * k + 2] = it.vertices[k][2];
					saveData[5 * k + 3] = it.u[k];
					saveData[5 * k + 4] = it.v[k];
				}
				writeVarToFile(saveData, convertedSavedModel);
			}
		}
	}

	return models;
}

struct BufferedData
{
	std::vector<uint8_t> data;
	size_t i;
	std::string sourcePath;

	BufferedData(std::string path)
	{
		FILE* f = fopen(path.c_str(), "rb");
		if (!f) throw std::runtime_error("Failed to open " + path);
		if (fseek(f, 0, SEEK_END)) throw std::runtime_error("fseek to end failed for " + path);
		size_t sz = _ftelli64(f);
		if (fseek(f, 0, SEEK_SET)) throw std::runtime_error("fseek to beg failed for " + path);

		data.resize(sz);
		i = 0;
		if (1 != fread(data.data(), sz, 1, f)) throw std::runtime_error("Invalid number of elements read by fread from " + path);
		fclose(f);
		sourcePath = path;
	}

	template<typename T>
	void read(T& value)
	{
		/*void readVarFromFile(T & var, std::ifstream & file, size_t sizeOverride = 0)
		{
			if (!sizeOverride) file.read((char*)(&var), sizeof(var));
			else file.read((char*)(&var), sizeOverride);
		}*/

		constexpr size_t sz = sizeof(T);
		int64_t available = data.size() - i;
		if (available < sz) throw std::runtime_error(std::string("Attempted to read ") + std::to_string(sz) + " bytes while only " + std::to_string(available) + " are available for " + sourcePath);

		memcpy(&value, data.data() + i, sz);
		i += sz;
	}

	bool eof()
	{
		return i >= data.size();
	}
};
std::vector<AssetLoader::ImportedModel> AssetLoader::loadBmdl(std::string path)
{
	std::vector<ImportedModel> ret;
	std::string parentDir = getFolderFromPath(path, true);

	float triangleData[15];
	BufferedData buf(path);
	while (!buf.eof())
	{
		uint64_t modelSize;
		buf.read(modelSize);
		uint64_t bytesRemaining = modelSize;
		if (modelSize % sizeof(triangleData) != 0) throw std::runtime_error("Error while loading model" + path + ": unexpected model size: " + std::to_string(modelSize) + " bytes, not mod " + std::to_string(sizeof(triangleData)));

		std::string textureRelPath;
		char c = -1;
		while (c != 0)
		{
			buf.read(c);
			textureRelPath.push_back(c);
		}
		std::string textureFullPath = parentDir + textureRelPath;

		std::vector<ImportedTriangle> tris;
		while (bytesRemaining > 0)
		{
			buf.read(triangleData);
			ImportedTriangle& t = tris.emplace_back();
			for (int i = 0; i < 3; ++i)
			{
				t.vertices[i][0] = triangleData[i * 5];
				t.vertices[i][1] = triangleData[i * 5+1] * -1;
				t.vertices[i][2] = triangleData[i * 5+2];
				t.u[i] = triangleData[i * 5 + 3];
				t.v[i] = triangleData[i * 5 + 4];
			}
			bytesRemaining -= sizeof(triangleData);
		}

		ImportedModel m;
		m.diffuseMapPath = textureFullPath;
		m.triangles = tris;
		ret.emplace_back(m);
	}

	return ret;
}
