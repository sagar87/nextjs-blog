import fs from "fs";
import Link from "next/link";

const getPostMetaData = () => {
  const folder = "posts/";
  const files = fs.readdirSync(folder);
  const markDownFiles = files.filter((file) => file.endsWith(".md"));
  const slugs = markDownFiles.map((file) => file.replace(".md", ""));

  return slugs;
};

const HomePage = () => {
  const slugs = getPostMetaData();
  const postPreview = slugs.map((slug) => (
    <div key={slug}>
      <Link href={`posts/${slug}`}>
        <h2>{slug}</h2>
      </Link>
    </div>
  ));
  return (
    <div>
      HomePage
      {postPreview}
    </div>
  );
};

export default HomePage;
