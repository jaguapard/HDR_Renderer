#pragma once
#include <vector>
#include "primitives.h"

namespace Rasterizing
{
	class ExplodeAndRestoreSceneEffect
	{
	public:
		ExplodeAndRestoreSceneEffect(double startTime, double explosionDuration, double restoreDuration, size_t triangleCount);
		void onFrameStart(double gameTime);

		//Applies this effect to triangles passed. Masked out triangles are passed through from input without change
		std::array<VertexPack16, 3> applyToTriangles(const std::array<VertexPack16, 3>& verts, const int32x16 triangleInd, Mask16 mask) const;
		//MatrixPack16_4x4 getTransformationMatrices(int32x16 vertInd, Mask16 mask) const;
		bool isFinished() const;
	private:
		double startTime, flipTime, endTime, gameTime;
		float effectProgress = 0;
		std::vector<float> shift[3], rot[3]; //shift and rotations for each axis, applied per-triangle
	};
}