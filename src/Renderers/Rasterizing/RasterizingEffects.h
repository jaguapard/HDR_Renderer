#pragma once
#include <vector>
#include "primitives.h"

namespace Rasterizing
{
	class ExplodeAndRestoreSceneEffect
	{
	public:
		ExplodeAndRestoreSceneEffect(double startTime, double explosionDuration, double restoreDuration, size_t triangleCount);

		//This should be called on each frame this effect is active.
		void onFrameStart(double gameTime);

		//Applies this effect to triangles passed. Masked out triangles are passed through from input without change
		std::array<VertexPack16, 3> applyToTriangles(const std::array<VertexPack16, 3>& verts, const int32x16 triangleInd, Mask16 mask) const;

		//Wheter or not this effect is finished. Using the effect past it's finish time is undefined behavior.
		bool isFinished() const;
	private:
		double startTime, flipTime, endTime, gameTime;
		float effectProgress = 0;
		std::vector<float> shift[3], rot[3]; //shift and rotations for each axis, applied per-triangle
	};
}