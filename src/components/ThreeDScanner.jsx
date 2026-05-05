import { useRef, useState } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { Float, PerspectiveCamera, Environment, ContactShadows, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';

/**
 * FingerprintAnimation component
 * Replaces the generic sphere with a high-tech, animated fingerprint visualization.
 * Features:
 * - 3D Distorted plane for an organic "biometric" feel
 * - Moving scan line effect
 * - Interactive hover states
 * - Modern holographic aesthetics
 */
function FingerprintAnimation() {
  const meshRef = useRef();
  const scanLineRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  // Load the fingerprint texture
  const texture = useLoader(THREE.TextureLoader, '/fingerprint-scanning-biometric-technology-GET-1492029015.2400.webp');

  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    
    if (meshRef.current) {
      // Subtle tilt based on mouse movement for 3D depth
      meshRef.current.rotation.y = THREE.MathUtils.lerp(meshRef.current.rotation.y, state.mouse.x * 0.3, 0.1);
      meshRef.current.rotation.x = THREE.MathUtils.lerp(meshRef.current.rotation.x, -state.mouse.y * 0.3, 0.1);
    }
    
    if (scanLineRef.current) {
      // Scanning line moving up and down across the fingerprint
      scanLineRef.current.position.y = Math.sin(time * 1.2) * 2.2;
    }
  });

  return (
    <group>
      <Float speed={2} rotationIntensity={0.2} floatIntensity={0.4}>
        {/* Main Holographic Fingerprint Mesh */}
        <mesh
          ref={meshRef}
          onPointerOver={() => setHovered(true)}
          onPointerOut={() => setHovered(false)}
          scale={hovered ? 1.05 : 1}
        >
          <planeGeometry args={[3.8, 4.5, 64, 64]} />
          <MeshDistortMaterial
            map={texture}
            transparent={true}
            opacity={0.85}
            color={hovered ? "#93c5fd" : "#3b82f6"}
            distort={0.15} 
            speed={1.5}
            metalness={0.6}
            roughness={0.2}
            emissive={hovered ? "#3b82f6" : "#1e40af"}
            emissiveIntensity={0.6}
            side={THREE.DoubleSide}
            alphaTest={0.05}
          />
        </mesh>

        {/* Animated Scan Line (Laser) */}
        <mesh ref={scanLineRef} position={[0, 0, 0.15]}>
          <planeGeometry args={[4.2, 0.08]} />
          <meshBasicMaterial 
            color="#60a5fa" 
            transparent={true} 
            opacity={0.9}
            blending={THREE.AdditiveBlending}
          />
          {/* Scan line glow effect */}
          <mesh position={[0, 0, -0.01]}>
            <planeGeometry args={[4.4, 0.4]} />
            <meshBasicMaterial 
              color="#3b82f6" 
              transparent={true} 
              opacity={0.3}
              blending={THREE.AdditiveBlending}
            />
          </mesh>
        </mesh>

        {/* Backlight/Halo for depth */}
        <mesh position={[0, 0, -0.4]}>
          <circleGeometry args={[3, 32]} />
          <meshBasicMaterial 
            color="#2563eb" 
            transparent={true} 
            opacity={0.1}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      </Float>
    </group>
  );
}

export default function ThreeDScanner() {
  return (
    <div className="w-full h-full min-h-[400px] relative">
      <Canvas shadows dpr={[1, 2]} camera={{ position: [0, 0, 6], fov: 45 }}>
        <ambientLight intensity={0.7} />
        <spotLight position={[10, 10, 10]} intensity={1.5} penumbra={1} />
        <pointLight position={[-10, -10, -10]} color="#3b82f6" intensity={1} />
        
        <FingerprintAnimation />
        
        <Environment preset="city" />
        <ContactShadows 
          position={[0, -3, 0]} 
          opacity={0.4} 
          scale={10} 
          blur={2} 
          far={4.5} 
        />
      </Canvas>
    </div>
  );
}
