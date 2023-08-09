//! Visualizes vertex placements based on sampling an implicit SDF on a uniform
//! grid. This is not intended to be a complete SDF contouring solution, just a
//! debug visualization for the vertex placement.

use bevy::prelude::*;
use bevy_polyline::{
    prelude::{Polyline, PolylineBundle, PolylineMaterial},
    PolylinePlugin,
};
use nalgebra::{Point3, Vector2, Vector3};
use qef::Qef3;
use smooth_bevy_cameras::{
    controllers::fps::{FpsCameraBundle, FpsCameraController, FpsCameraPlugin},
    LookTransformPlugin,
};
use std::ops::Range;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LookTransformPlugin)
        .add_plugin(FpsCameraPlugin::default())
        .add_plugin(PolylinePlugin)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
) {
    let sdf = |p: Point3<f32>| {
        implicit::torus(Vector2::new(4.0, 2.0), p)
            .max(implicit::plane(
                Vector3::zeros(),
                Vector3::new(1.0, 1.0, 1.0).normalize(),
                p,
            ))
            .max(-implicit::torus(Vector2::new(4.0, 1.5), p))
    };

    // Estimate a vertex for each bipolar cell.
    let mut vertex_estimates = Vec::new();
    let mut edge_crossings = Vec::new();
    let bounds_min = Point3::new(-10.0, -10.0, -10.0);
    let bounds_max = Point3::new(10.0, 10.0, 10.0);
    let resolution = 1 << 7;
    let cell_size = (bounds_max - bounds_min) / resolution as f32;
    info!("cell size = {cell_size:?}");
    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                let cell_min = bounds_min
                    + Vector3::new(x as f32, y as f32, z as f32).component_mul(&cell_size);
                let cell_max = cell_min + cell_size;
                if let Some(e) =
                    CellVertexEstimate::new(cell_min, cell_max, &sdf, &mut edge_crossings)
                {
                    vertex_estimates.push(e);
                }
            }
        }
    }
    info!("# vertices = {}", vertex_estimates.len());

    render_cell_vertex_estimates(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut polylines,
        &mut polyline_materials,
        &vertex_estimates,
        &edge_crossings,
    );

    commands
        .spawn(Camera3dBundle::default())
        .insert(FpsCameraBundle::new(
            FpsCameraController {
                translate_sensitivity: 3.0,
                ..Default::default()
            },
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::ZERO,
            Vec3::Y,
        ));
}

fn render_cell_vertex_estimates(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    polylines: &mut Assets<Polyline>,
    polyline_materials: &mut Assets<PolylineMaterial>,
    vertex_estimates: &[CellVertexEstimate],
    edge_crossings: &[EdgeCrossing],
) {
    let sphere_mesh = meshes.add(Mesh::from(shape::UVSphere {
        radius: 0.01,
        sectors: 3,
        stacks: 3,
    }));

    // let edge_crossing_material = materials.add(StandardMaterial {
    //     unlit: true,
    //     ..StandardMaterial::from(Color::RED)
    // });
    let surface_sphere_material = materials.add(StandardMaterial {
        unlit: true,
        ..StandardMaterial::from(Color::RED)
    });
    // let cell_edge_material = polyline_materials.add(PolylineMaterial {
    //     width: 2.0,
    //     color: Color::BLUE,
    //     ..default()
    // });
    let surface_line_material = polyline_materials.add(PolylineMaterial {
        width: 1.0,
        color: Color::GREEN,
        ..default()
    });

    // Batched for less overhead.
    let mut cell_edge_polyline = Polyline::default();
    let mut surface_polyline = Polyline::default();

    for estimate in vertex_estimates {
        // // Vertex estimate.
        commands.spawn(MaterialMeshBundle::<StandardMaterial> {
            mesh: sphere_mesh.clone(),
            material: surface_sphere_material.clone(),
            transform: Transform::from_translation(estimate.vertex.into()),
            ..default()
        });

        // Edge crossings.
        for i in estimate.edge_crossings_range.clone() {
            let crossing = &edge_crossings[i];

            // commands.spawn(MaterialMeshBundle::<StandardMaterial> {
            //     mesh: sphere_mesh.clone(),
            //     material: edge_crossing_material.clone(),
            //     transform: Transform::from_translation(estimate.vertex.into()),
            //     ..default()
            // });

            // NAN hack that lets us combine lines with jump discontinuities.
            cell_edge_polyline.vertices.extend([
                crossing.edge_start.into(),
                crossing.edge_end.into(),
                Vec3::NAN,
            ]);
            surface_polyline.vertices.extend([
                crossing.crossing.into(),
                estimate.vertex.into(),
                Vec3::NAN,
            ]);
        }
    }

    // commands.spawn(PolylineBundle {
    //     polyline: polylines.add(cell_edge_polyline),
    //     material: cell_edge_material.clone(),
    //     ..default()
    // });
    commands.spawn(PolylineBundle {
        polyline: polylines.add(surface_polyline),
        material: surface_line_material.clone(),
        ..default()
    });
}

struct CellVertexEstimate {
    vertex: Point3<f32>,
    // Edge crossings are kept in a global buffer to reduce allocations.
    edge_crossings_range: Range<usize>,
}

struct EdgeCrossing {
    edge_start: Point3<f32>,
    edge_end: Point3<f32>,
    crossing: Point3<f32>,
}

impl CellVertexEstimate {
    fn new(
        cell_min: Point3<f32>,
        cell_max: Point3<f32>,
        sdf: impl Fn(Point3<f32>) -> f32,
        edge_crossings: &mut Vec<EdgeCrossing>,
    ) -> Option<Self> {
        let corners = cell_corners(cell_min, cell_max);
        let samples = corners.map(&sdf);

        if !cell_is_bipolar(&samples) {
            return None;
        }

        // Calculate edge crossings and their centroid.
        let mut centroid = Vector3::zeros();
        let edge_crossings_start = edge_crossings.len();
        let mut num_edge_crossings = 0;
        for [e1, e2] in CELL_EDGES {
            let s1 = samples[e1];
            let s2 = samples[e2];
            if (s1 < 0.0) != (s2 < 0.0) {
                // Lerp the edge vertices.
                let diff = s2 - s1;
                let s1_lerp = s2 / diff;
                let s2_lerp = -s1 / diff;
                let crossing = s1_lerp * corners[e1].coords + s2_lerp * corners[e2].coords;
                num_edge_crossings += 1;
                centroid += crossing;
                edge_crossings.push(EdgeCrossing {
                    edge_start: corners[e1],
                    edge_end: corners[e2],
                    crossing: crossing.into(),
                });
            }
        }
        centroid /= num_edge_crossings as f32;
        let centroid = Point3::from(centroid);
        let edge_crossings_range =
            edge_crossings_start..(edge_crossings_start + num_edge_crossings);

        // Calculate QEF of edge crossings relative to centroid.
        let mut qef = Qef3::zeros();
        for i in edge_crossings_range.clone() {
            let p = edge_crossings[i].crossing;
            let normal = central_gradient(&sdf, p, 0.0001).normalize();
            let rel_p = Point3::from(p - centroid);
            qef += Qef3::probabilistic_plane(rel_p, normal, 0.01, 0.01);
            // qef += Qef3::plane(rel_p, normal);
        }

        // let vertex = centroid + qef.minimizer_with_pseudo_inverse(0.1);
        let vertex = centroid + qef.minimizer_with_exact_inverse();

        Some(Self {
            vertex,
            edge_crossings_range,
        })
    }
}

fn cell_is_bipolar(samples: &[f32; 8]) -> bool {
    let mut any_negative = false;
    let mut any_positive = false;
    for &sample in samples {
        any_negative |= sample < 0.0;
        any_positive |= sample >= 0.0;
    }
    any_negative && any_positive
}

// Central differencing around `p`. Needs 6 samples.
fn central_gradient(sdf: impl Fn(Point3<f32>) -> f32, p: Point3<f32>, delta: f32) -> Vector3<f32> {
    let h = 0.5 * delta;
    let dx = Vector3::new(h, 0.0, 0.0);
    let dy = Vector3::new(0.0, h, 0.0);
    let dz = Vector3::new(0.0, 0.0, h);
    Vector3::new(
        sdf(p + dx) - sdf(p - dx),
        sdf(p + dy) - sdf(p - dy),
        sdf(p + dz) - sdf(p - dz),
    ) / delta
}

fn cell_corners(min: Point3<f32>, max: Point3<f32>) -> [Point3<f32>; 8] {
    [
        Point3::new(min.x, min.y, min.z),
        Point3::new(max.x, min.y, min.z),
        Point3::new(min.x, max.y, min.z),
        Point3::new(max.x, max.y, min.z),
        Point3::new(min.x, min.y, max.z),
        Point3::new(max.x, min.y, max.z),
        Point3::new(min.x, max.y, max.z),
        Point3::new(max.x, max.y, max.z),
    ]
}

const CELL_EDGES: [[usize; 2]; 12] = [
    [0b000, 0b001],
    [0b000, 0b010],
    [0b000, 0b100],
    [0b001, 0b011],
    [0b001, 0b101],
    [0b010, 0b011],
    [0b010, 0b110],
    [0b100, 0b101],
    [0b100, 0b110],
    [0b110, 0b111],
    [0b101, 0b111],
    [0b011, 0b111],
];

mod implicit {
    use nalgebra::{Point3, Vector2, Vector3};

    pub fn plane(o: Vector3<f32>, n: Vector3<f32>, p: Point3<f32>) -> f32 {
        (p - o).coords.dot(&n)
    }

    pub fn torus(t: Vector2<f32>, p: Point3<f32>) -> f32 {
        let q = Vector2::new(p.xz().coords.magnitude() - t.x, p.y);
        q.magnitude() - t.y
    }
}
