import fs from "fs";
import Markdown from "markdown-to-jsx";

const getPageContent = (slug: string) => {
  const folder = "posts";
  const file = `${folder}/${slug}.md`;
  const content = fs.readFileSync(file, "utf-8");
  return content;
};

const PostPage = async (props: any) => {
  const { slug } = await props.params;
  const content = getPageContent(slug);
  return (
    <div>
      <h2>{slug}</h2>
      <Markdown>{content}</Markdown>
    </div>
  );
};

export default PostPage;
