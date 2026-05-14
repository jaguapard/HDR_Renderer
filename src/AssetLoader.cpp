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
	if (convertToSavePath.length() > 0)
	{
		convertedSavedModel = std::ofstream(convertToSavePath, std::ios::binary);
		uint64_t version = 1;
		writeVarToFile(version, convertedSavedModel);
	}

	for (size_t i = 0; i < pScene->mNumMeshes; i++)
	{
		aiMesh* mesh = pScene->mMeshes[i];
		aiMaterial* material = pScene->mMaterials[mesh->mMaterialIndex];
		aiString texturePath;
		material->GetTexture(aiTextureType_DIFFUSE, 0, &texturePath);
		//material->GetTexture(aiTex, 0, &texturePath);

		std::string textureRelPath = texturePath.C_Str();
		std::string textureFullPath = getFolderFromPath(path, true) + textureRelPath;

		std::vector<AssetLoader::ImportedTriangle> tris;
		if (mesh->mTextureCoordsNames)
		{
			for (int j = 0; j < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++j)
			{
				aiString* coordName = mesh->mTextureCoordsNames[j];
				if (!coordName) continue;
				std::cout << "Texture coord " << j << " for model " << i << " in "  << path << " is named: " << coordName->C_Str() << "\n";
			}
		}
		//else std::cout << "Texture coords names for model " << i << " in " << path << " are not available.\n";
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
				aiVector3D aiNormal = mesh->mNormals[vertIndex];

				auto v = aiToBob(aiVertice);
				t.v[k].space = aiVertice;
				t.v[k].diffuseMapCoords = aiUVs;
				t.v[k].normal = aiNormal;
			}
		}

		ImportedModel model;
		model.diffuseMapPath = textureFullPath;
		model.triangles = tris;
		models.emplace_back(model);

		//if this is true, will save BMDL to path. This format is much lighter and fast to import
		if (convertToSavePath.length() > 0)
		{
			uint64_t modelSize = tris.size() * sizeof(ImportedTriangle); //size of a single model entry in bytes, not counting size member and path. 
			writeVarToFile(modelSize, convertedSavedModel);
			uint32_t texturePathLen = textureRelPath.length();
			writeVarToFile(texturePathLen, convertedSavedModel);
			convertedSavedModel.write(textureRelPath.c_str(), texturePathLen);
			for (const auto& it : models.back().triangles)
			{
				writeVarToFile(it, convertedSavedModel);
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

	BufferedData buf(path);
	uint64_t version;
	buf.read(version);
	if (version != 1) throw std::runtime_error("Unsupported version for BMDL file: " + std::to_string(version));

	while (!buf.eof())
	{
		uint64_t modelSize;
		buf.read(modelSize);
		uint64_t bytesRemaining = modelSize;
		if (modelSize % sizeof(ImportedVertex) != 0) throw std::runtime_error("Error while loading model" + path + ": unexpected model size: " + std::to_string(modelSize) + " bytes, not mod " + std::to_string(sizeof(ImportedTriangle)));

		std::string textureRelPath;
		uint32_t texturePathLen;
		buf.read(texturePathLen);
		for (int i = 0; i < texturePathLen; ++i)
		{
			char c;
			buf.read(c);
			textureRelPath.push_back(c);
		}
		std::string textureFullPath = parentDir + textureRelPath;

		std::vector<ImportedTriangle> tris;
		while (bytesRemaining > 0)
		{
			ImportedTriangle& t = tris.emplace_back();
			buf.read(t);
			bytesRemaining -= sizeof(t);
		}

		ImportedModel& m = ret.emplace_back();
		m.diffuseMapPath = textureFullPath;
		m.triangles = std::move(tris);
	}

	return ret;
}
